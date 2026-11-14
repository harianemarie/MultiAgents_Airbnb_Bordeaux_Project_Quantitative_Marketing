"""Real execution events and explicit manager gates for the notebook agents."""
from datetime import datetime, timezone
import functools
import time

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
        self.file = None

    def _install(self):
        runtime, ns = self, self.ns
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
            answer = original_llm(*args, **kwargs)
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
                    result = original(*args, **kwargs)
                    if name == 'find_comparables': summary=f"{result['n_comparables']} comparables; median €{result['median']}; radius {result['radius_km']} km."
                    elif name == 'predict_price_range': summary=f"Estimate €{result['predicted_price']}; 80% interval {result['range_80pct']}."
                    elif name == 'summarise_reviews': summary=f"{len(result.get('themes', {}))} themes; {len(result.get('excerpts', []))} sourced excerpts."
                    elif name == 'check_availability': summary=f"{result.get('n_comparables_free', 'No')} competitors free on the requested dates."
                    else: summary='Structured property context ready.'
                    runtime.emit('tool_result', actor, summary, tool=name, source=result.get('source', 'Manager property brief'), seconds=round(time.perf_counter()-started, 3))
                    return result
                return tool
            ns[name] = make_tool(original, name, description)
        ns['ANALYST_TOOLS'] = [ns[k] for k in descriptions]

    def stage(self, actor, operation, file):
        self.active = actor
        self.states[actor] = 'running'
        started = time.perf_counter()
        self.emit('start', actor, self.ns['AGENT_CONFIG'][actor]['input'])
        try:
            result = operation(file)
        except Exception:
            self.states[actor] = 'error'
            self.emit('error', actor, 'This stage failed. No manager approval was granted.')
            self.active = None
            raise
        self.durations[actor] = round(time.perf_counter()-started, 2)
        self.states[actor] = 'done'
        # This describes the result actually selected by the agent, not merely whether a key exists.
        logs = [l for l in file.data['log'] if l.startswith('['+actor)]
        final_log = logs[-1] if logs else ''
        if '(LLM)' in final_log: mode = 'Gemini'
        elif '(fallback)' in final_log: mode = 'Rules'
        elif actor == 'Validator': mode = 'Rules + Gemini' if file.data.get('validation', {}).get('llm_review') else 'Rules'
        else: mode = 'Python tools'
        self.modes[actor] = mode
        self.emit('complete', actor, self.ns['AGENT_CONFIG'][actor]['output'], seconds=self.durations[actor], mode=mode)
        self.active = None
        return result

    def analyse(self, args, brand_brief, objective):
        self.reset()
        info, padded, request = self.ns['form_to_info'](*args)
        dates = (info['checkin'], info['checkout']) if info['checkin'] and info['checkout'] else None
        prop = self.ns['build_property'](info)
        prop['amenities'] = self.ns['json'].dumps(padded)
        file = self.ns['SharedPropertyFile'](prop, dates, request=request)
        file.verbose = False
        file.data.update(brand_brief=brand_brief, objective=objective, strategy_approved=False, final_approved=False,
                         strategy_version=1, studio_events=self.events, revision_count=0)
        file.data['highlights'] = self.ns['split_highlights'](args[8])
        file.data['warnings'] = self.ns['size_warnings'](info['accommodates'], info['bedrooms'], info['beds'], info['bathrooms'])
        self.file = file
        self.emit('brief', 'Manager', 'Property brief and brand context submitted. Objective: '+objective)
        self.stage('Planner', self.ns['planner'].prepare, file)
        self.emit('handoff', 'Planner → Analyst', 'Property features and dates are available in the shared context.')
        self.stage('Analyst', self.ns['analyst'].run, file)
        self.emit('handoff', 'Analyst → Strategist', 'Sending comparables, price interval and sourced guest insights.')
        self.stage('Strategist', self.ns['strategist'].run, file)
        self.states['Creative'] = 'approval'
        self.emit('approval', 'Manager', 'Review segmentation, target, positioning and price before creative execution.')
        return file

    def approve_strategy(self, file, target, price):
        if not target.strip() or price <= 0:
            raise ValueError('Provide a target and a positive nightly price.')
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

    def create_and_validate(self, file):
        if not file.data.get('strategy_approved'):
            raise ValueError('Manager approval of the current strategy is required.')
        while True:
            self.stage('Creative', self.ns['creative'].run, file)
            self.emit('handoff', 'Creative → Validator', 'Checking the listing and campaign against the evidence and approved strategy.')
            self.stage('Validator', self.ns['validator'].run, file)
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
        return file

    def revise_strategy(self, file):
        file.data.setdefault('revision_history', []).append(dict(strategy=dict(file.data['strategy']), validation=file.data.get('validation', {})))
        file.data['strategy_approved'] = False
        file.data['final_approved'] = False
        file.data['strategy_version'] += 1
        self.stage('Strategist', self.ns['strategist'].run, file)
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
        if not file.data.get('strategy_approved') or file.data.get('validation', {}).get('status') != 'VALIDATED':
            raise ValueError('The current strategy must be approved and all validation issues resolved.')
        if file.data.get('final_approved'): return
        file.data['final_approved'] = True
        file.data['approvals'].append(dict(step='final listing', decision='approved', by='manager',
                                            version=file.data['strategy_version'], timestamp=datetime.now(timezone.utc).isoformat()))
        self.ns['MEMORY'].save(file)
        self.emit('approval', 'Manager', 'Final listing and campaign approved for export. Nothing has been posted to Airbnb.')
