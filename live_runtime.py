"""Real execution events and explicit manager gates for the notebook agents."""
from datetime import datetime, timezone
import functools
import time
import copy
from contextlib import nullcontext
from quality import response_issues, stage_issues, blocking, source_registry, QualityError

AGENTS = ('Planner', 'Analyst', 'Strategist', 'Creative', 'Validator')

class StudioRuntime:
    def __init__(self, namespace):
        self.ns = namespace
        self.events = []
        self.states = {a: 'waiting' for a in AGENTS}
        self.durations = {}
        self.modes = {}
        self.active = None
        self.start_time = time.perf_counter()
        self.on_update = None
        self.file = None
        self._price_cache = {}
        self.usage = []
        self.audit = []
        self.failure = None
        self._pending = None
        self._checkpoint = None
        self._install()

    def emit(self, kind, actor, message, **extra):
        event = dict(kind=kind, actor=actor, message=str(message),
                     elapsed=round(time.perf_counter()-self.start_time, 2),
                     timestamp=datetime.now(timezone.utc).isoformat(), **extra)
        self.events.append(event)
        if self.on_update: self.on_update(self)
        return event

    def reset(self):
        self.events.clear()
        self.states = {a: 'waiting' for a in AGENTS}
        self.durations.clear()
        self.modes.clear()
        self.start_time = time.perf_counter()
        self._price_cache.clear()
        self.usage.clear()
        self.audit.clear()
        self.failure = None
        self._pending = None
        # Keep the previous results until the new property has been prepared.

    def _install(self):
        runtime, ns = self, self.ns
        ns['_usage_event'] = lambda record: runtime.usage.append(dict(record, actor=runtime.active or 'System'))
        ns['_network_event'] = lambda kind, model, message: runtime.emit(kind, runtime.active or 'System', message, model=model)
        original_write = ns['SharedPropertyFile'].write
        def write(file, actor, message):
            original_write(file, actor, message)
            runtime.emit('message', actor, message)
        ns['SharedPropertyFile'].write = write
        original_llm = ns['ask_llm']
        @functools.wraps(original_llm)
        def llm(*args, **kwargs):
            actor = runtime.active or 'Planner'
            if ns.get('llm_client') is None:
                runtime.emit('fallback', actor, 'Gemini is off. Applying the explicit rule-based method.')
                return original_llm(*args, **kwargs)
            runtime.emit('llm', actor, 'Request sent to Gemini; waiting for a structured response.', model=ns['GEMINI_MODEL'])
            args=list(args)
            if args and runtime.file is not None and actor == 'Analyst':
                registry=runtime.file.data.get('sources',[])
                args[0]+='\nFor every key fact, copy one of these source ids or references exactly: '+ns['json'].dumps(registry)
            answer = original_llm(*args, **kwargs)
            problems=response_issues(actor,answer) if answer is not None else []
            if actor == 'Analyst' and answer is not None and blocking(problems):
                # One bounded formatting repair; never rerun tools or infer missing evidence.
                runtime.emit('llm_repair',actor,'Requesting one correction of the Analyst JSON format; evidence and source requirements remain unchanged.')
                repair_system = args[0] + '\nFORMAT REPAIR ONLY: Return the corrected JSON object. Preserve all factual claims and numbers from the draft. Do not add facts or invent a missing source. Use a registry ID only where the existing draft explicitly identifies the matching source. If a claim has no identifiable source, omit that claim. Do not call tools.'
                repair_prompt = ns['json'].dumps(dict(draft=answer,errors=[p['message'] for p in blocking(problems)]),default=str)
                answer = original_llm(repair_system,repair_prompt,temperature=0,max_tokens=4096)
                problems=response_issues(actor,answer) if answer is not None else []
            if blocking(problems):
                runtime.emit('invalid_response',actor,'Structured response rejected: '+'; '.join(p['message'] for p in blocking(problems)))
                answer=None
            runtime.emit('llm_result' if answer else 'fallback', actor,
                         'Structured response received.' if answer else 'No usable response. Applying the rule-based method.',
                         model=ns.get('ACTIVE_MODEL'))
            return answer
        ns['ask_llm'] = llm
        descriptions = {
            'get_property_profile': 'Reading the shared property profile',
            'find_comparables': 'Searching and ranking comparable properties',
            'predict_price_range': 'Predicting price, uncertainty and SHAP drivers',
            'check_availability': 'Checking calendar availability for the selected stay',
            'summarise_reviews': 'Comparing guest-review themes with the Bordeaux market',
        }
        for name, description in descriptions.items():
            original = ns[name]
            def make_tool(original, name, description):
                @functools.wraps(original)
                def tool(*args, **kwargs):
                    if runtime.active is None:
                        return original(*args, **kwargs)
                    actor = runtime.active
                    runtime.emit('tool', actor, description, tool=name)
                    started = time.perf_counter()
                    if name == 'predict_price_range':
                        key = ns['CURRENT']['x_row'].to_json(orient='split')
                        if key not in runtime._price_cache:
                            # Model resources are shared; network calls and agent state are not.
                            with ns.get('_model_lock', nullcontext()):
                                runtime._price_cache[key] = original(*args, **kwargs)
                        result = copy.deepcopy(runtime._price_cache[key])
                    else:
                        result = original(*args, **kwargs)
                    if name == 'find_comparables': summary=f"{result['n_comparables']} comparables; median €{result['median']}; radius {result['radius_km']} km."
                    elif name == 'predict_price_range': summary=f"Estimate €{result['predicted_price']}; 80% interval {result['range_80pct']}."
                    elif name == 'summarise_reviews': summary=f"{len(result.get('themes', {}))} themes; {len(result.get('excerpts', []))} sourced excerpts."
                    elif name == 'check_availability': summary=(f"{result['n_comparables_free']} competitors free on the requested dates." if result.get('dates') else 'No stay dates supplied: availability was not assessed.')
                    else: summary='Structured property context ready.'
                    runtime.emit('tool_result', actor, summary, tool=name, source=result.get('source', 'Not applicable: no dates' if name=='check_availability' and not result.get('dates') else 'Manager property brief'), seconds=round(time.perf_counter()-started, 3))
                    return result
                return tool
            ns[name] = make_tool(original, name, description)
        ns['ANALYST_TOOLS'] = [ns[k] for k in descriptions]

    def stage(self, actor, operation, file):
        # Latest checkpoint only, kept in this browser session, before the stage mutates state.
        snapshot=copy.deepcopy({k:v for k,v in file.data.items() if k not in ('studio_events','quality_checks')})
        self._checkpoint=dict(data=snapshot,current=dict(self.ns['CURRENT']),states=dict(self.states),modes=dict(self.modes))
        self.active=actor
        self.states[actor]='running'
        started=time.perf_counter()
        self.emit('start',actor,self.ns['AGENT_CONFIG'][actor]['input'])
        try:
            result=operation(file)
            problems=stage_issues(actor,file.data)
            record=dict(stage=actor,passed=not bool(blocking(problems)),issues=problems,
                        timestamp=datetime.now(timezone.utc).isoformat())
            self.audit.append(record)
            file.data['quality_checks']=self.audit
            self.emit('quality_check',actor,'Intermediate checks passed.' if record['passed'] else 'Intermediate checks blocked this output.',issues=problems)
            if blocking(problems):raise QualityError('; '.join(p['message'] for p in blocking(problems)))
        except Exception as exc:
            file.data.clear();file.data.update(copy.deepcopy(snapshot))
            file.data['studio_events']=self.events;file.data['quality_checks']=self.audit
            self.ns['CURRENT'].clear();self.ns['CURRENT'].update(self._checkpoint['current'])
            self.failure=dict(actor=actor,error_type=type(exc).__name__,message=str(exc)[:250],pending=copy.deepcopy(self._pending))
            self.states[actor]='error'
            self.active=None
            self.durations[actor]=self.durations.get(actor,0)+round(time.perf_counter()-started,2)
            self.emit('error',actor,'Stage failed. The previous successful stages are retained; Retry resumes here.',error_type=type(exc).__name__)
            raise
        self.failure=None
        self.durations[actor]=self.durations.get(actor,0)+round(time.perf_counter()-started,2)
        self.states[actor]='done'
        logs=[line for line in file.data['log'] if line.startswith('['+actor)]
        final_log=logs[-1] if logs else ''
        if '(LLM)' in final_log:mode='Gemini'
        elif '(fallback)' in final_log:mode='Rules'
        elif actor=='Validator':mode='Rules + Gemini' if file.data.get('validation',{}).get('llm_review') else 'Rules'
        else:mode='Python tools'
        self.modes[actor]=mode
        self.active=None
        self.emit('complete',actor,self.ns['AGENT_CONFIG'][actor]['output'],seconds=self.durations[actor],mode=mode)
        return result

    def retry(self):
        if not self.failure or not self.file:raise ValueError('There is no failed stage to resume.')
        pending=self.failure.get('pending')
        if not pending:raise ValueError('This operation has no resumable workflow.')
        actor=self.failure['actor'];self.failure=None
        self.emit('retry','Manager','Resuming '+actor+' from the last session checkpoint.')
        if pending['workflow']=='analysis':return self._continue_analysis(self.file,pending['actor'])
        if pending['workflow']=='creation':return self.create_and_validate(self.file,resume_actor=pending['actor'])
        if pending['workflow']=='strategy_revision':return self._finish_strategy_revision(self.file)
        raise ValueError('Unknown checkpoint workflow.')

    def analyse(self, args, brand_brief, objective):
        self.reset()
        info, padded, request = self.ns['form_to_info'](*args)
        dates = (info['checkin'], info['checkout']) if info['checkin'] and info['checkout'] else None
        prop = self.ns['build_property'](info)
        prop['amenities'] = self.ns['json'].dumps(padded)
        file = self.ns['SharedPropertyFile'](prop, dates, request=request)
        file.verbose = False
        file.data.update(brand_brief=brand_brief, objective=objective, strategy_approved=False, final_approved=False,
                         strategy_version=1, studio_events=self.events, revision_count=0, sources=source_registry(self.ns),quality_checks=self.audit)
        file.data['highlights'] = self.ns['split_highlights'](args[8])
        file.data['warnings'] = self.ns['size_warnings'](info['accommodates'], info['bedrooms'], info['beds'], info['bathrooms'])
        self.file = file
        self.emit('brief', 'Manager', 'Property brief and brand context submitted. Objective: '+objective)
        return self._continue_analysis(file)

    def _continue_analysis(self,file,next_actor='Planner'):
        sequence=['Planner','Analyst','Strategist']
        for actor in sequence[sequence.index(next_actor):]:
            self._pending=dict(workflow='analysis',actor=actor)
            operation=self.ns['planner'].prepare if actor=='Planner' else self.ns[actor.lower()].run
            self.stage(actor,operation,file)
            if actor=='Planner':self.emit('handoff','Planner → Analyst','Property features and dates are available in the shared context.')
            if actor=='Analyst':self.emit('handoff','Analyst → Strategist','Intermediate checks passed; forwarding sourced evidence.')
        self.states['Creative']='approval'
        self._pending=None
        self.emit('approval','Manager','Review the checked strategy before creative execution.')
        return file

    def approve_strategy(self, file, target, price):
        if not target.strip() or price <= 0:
            raise ValueError('Provide a target and a positive nightly price.')
        previous=copy.deepcopy(file.data['strategy'])
        file.data['strategy'].update(target=target.strip(),recommended_price=float(price))
        problems=stage_issues('Strategist',file.data)
        if blocking(problems):
            file.data['strategy']=previous
            raise QualityError('; '.join(p['message'] for p in blocking(problems)))
        if self.failure:raise ValueError('Retry the failed stage before approving.')
        self.emit('approval', 'Manager', 'Strategy approved for version '+str(file.data['strategy_version']))
        file.data['strategy']['target'] = target.strip()
        file.data['strategy']['recommended_price'] = float(price)
        file.data['strategy_approved'] = True
        file.data['final_approved'] = False
        file.data['approvals'].append(dict(step='strategy', decision='approved', by='manager',
                                            version=file.data['strategy_version'], target=target.strip(), price=float(price),
                                            timestamp=datetime.now(timezone.utc).isoformat()))
        self.emit('handoff', 'Strategist → Creative', 'The manager-approved strategy is now the creative brief.')
        return self.create_and_validate(file)

    def create_and_validate(self, file, resume_actor='Creative'):
        if not file.data.get('strategy_approved'):
            raise ValueError('Manager approval of the current strategy is required.')
        while True:
            if resume_actor!='Validator':
                self._pending=dict(workflow='creation',actor='Creative')
                self.stage('Creative', self.ns['creative'].run, file)
                self.emit('handoff', 'Creative → Validator', 'Checking the listing and campaign against the evidence and approved strategy.')
            self._pending=dict(workflow='creation',actor='Validator')
            self.stage('Validator', self.ns['validator'].run, file)
            resume_actor='Creative'
            issues = file.data['validation'].get('issues', [])
            if not issues:
                self.states['Validator'] = 'done'
                self.emit('approval', 'Manager', 'Validation passed. The final listing awaits your approval.')
                break
            self.states['Validator'] = 'review'
            if file.data['revision_count'] >= self.ns['MAX_REVISIONS']:
                self.emit('review', 'Validator', 'Revision limit reached. Review the issues and send explicit instructions to an agent.')
                break
            file.data['revision_count'] += 1
            redo = {i['agent'] for i in issues}
            file.data['feedback'] = {a:[i['issue'] for i in issues if i['agent']==a] for a in redo}
            self.emit('revision', 'Validator → '+', '.join(sorted(redo)), '; '.join(i['issue'] for i in issues))
            if 'Strategist' in redo:
                self.revise_strategy(file)
                break
            # Creative-only corrections keep the same approved strategy.
        self._pending=None
        return file

    def revise_strategy(self, file):
        file.data.setdefault('revision_history', []).append(dict(strategy=dict(file.data['strategy']), validation=file.data.get('validation', {})))
        file.data['strategy_approved'] = False
        file.data['final_approved'] = False
        file.data['strategy_version'] += 1
        return self._finish_strategy_revision(file)

    def _finish_strategy_revision(self,file):
        self._pending=dict(workflow='strategy_revision',actor='Strategist')
        self.stage('Strategist', self.ns['strategist'].run, file)
        self._pending=None
        file.data['creative'] = {}
        file.data['validation'] = {}
        self.states['Creative'] = 'approval'
        self.states['Validator'] = 'waiting'
        self.emit('approval', 'Manager', 'The strategy changed. Approve the new version before the Creative agent continues.')

    def request_revision(self, file, actor, feedback):
        if not feedback.strip(): raise ValueError('Describe what the agent should revise.')
        file.data['feedback'][actor] = [feedback.strip()]
        file.data['revision_count'] = 0
        file.data['final_approved'] = False
        self.emit('revision', 'Manager → '+actor, feedback.strip())
        if actor == 'Strategist': self.revise_strategy(file)
        else: self.create_and_validate(file)

    def approve_final(self, file):
        if self.failure:raise ValueError('Retry the failed stage before approving.')
        if not file.data.get('strategy_approved') or file.data.get('validation', {}).get('status') != 'VALIDATED':
            raise ValueError('The current strategy must be approved and all validation issues resolved.')
        if file.data.get('final_approved'): return
        file.data['final_approved'] = True
        file.data['approvals'].append(dict(step='final listing', decision='approved', by='manager',
                                            version=file.data['strategy_version'], timestamp=datetime.now(timezone.utc).isoformat()))
        self.ns['MEMORY'].save(file)
        self.emit('approval', 'Manager', 'Final listing and campaign approved for export. Nothing has been posted to Airbnb.')
