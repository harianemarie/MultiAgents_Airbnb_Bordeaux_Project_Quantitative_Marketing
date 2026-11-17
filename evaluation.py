"""Controlled-information comparison; no real API call unless explicitly requested."""
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import time
import copy
import csv
import joblib
from quality import source_registry, stage_issues, blocking, unsupported_amenity_flags, response_issues, QualityError
from live_runtime import StudioRuntime

PROPOSITION='Help short-term rental management companies prepare a new property for the market by connecting market analysis, positioning and listing creation in a workflow controlled by the property manager.'

def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if hasattr(value,'item'):
        try:return clean_json(value.item())
        except ValueError:pass
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(value)

def canonical(value):return json.dumps(clean_json(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def load_assets(root):
    root=Path(root);manifest=json.loads((root/'assets/manifest.json').read_text())
    blob=b''.join((root/'assets'/name).read_bytes() for name in manifest['parts'])
    if hashlib.sha256(blob).hexdigest()!=manifest['sha256']:raise ValueError('Asset checksum mismatch.')
    assets=joblib.load(BytesIO(blob))
    from review_nlp import require_verified_nlp
    require_verified_nlp(assets)
    import shap
    assets['explainer']=shap.TreeExplainer(assets['tree_for_shap'])
    return assets,manifest['sha256']

def namespace(root,assets,api_key='',model='gemini-2.5-flash'):
    ns=dict(assets,__name__='benchmark_engine',_api_key=api_key,_model=model)
    exec(compile((Path(root)/'engine_source.py').read_text(encoding='utf-8'),'engine_source.py','exec'),ns)
    # No silent model change within this comparison.
    ns['GEMINI_MODELS']=[model]
    ns['MEMORY'].cases=[]
    for cfg in ns['AGENT_CONFIG'].values():cfg['llm']['temperature']=0.3
    return ns

def cases(ns):
    nb=ns['NEIGHBOURHOODS'][0]
    distant=ns['NB_CENTRES'].copy()
    distance=ns['haversine_km'](distant.latitude,distant.longitude,*ns['BORDEAUX_CENTRE'])
    far=distance.idxmax()
    base=dict(neighbourhood=nb,room_type='Entire home/apt',accommodates=4,bedrooms=2,beds=2,bathrooms=1,surface_m2=55,description='Open kitchen.')
    scenarios=[
        ('family','Family apartment',{},['Wifi','Kitchen','Washer','Heating','Smoke alarm','Crib'],'Launch an honest family-oriented listing.','No special instruction.'),
        ('couple','Compact studio',dict(accommodates=2,bedrooms=0,beds=1,surface_m2=25),['Wifi','Kitchen','Heating','Smoke alarm'],'Position a compact city property.','Do not invent a separate bedroom or extra rooms.'),
        ('work_no_wifi','Professional target without equipment',dict(accommodates=2,bedrooms=1,beds=1),['Kitchen','Heating','Smoke alarm'],'The manager requests business travellers.','Wifi and a workspace are not confirmed. Challenge the requested target if necessary.'),
        ('sparse','Sparse property information',dict(accommodates=2,bedrooms=1,beds=1,surface_m2=None,description=''),[],'Create a cautious listing with limited information.','Do not infer missing equipment, reviews, views or services.'),
        ('atypical','Large property away from the centre',dict(neighbourhood=far,accommodates=12,bedrooms=6,beds=8,bathrooms=3,surface_m2=220),['Kitchen','Wifi','Washer','Free parking on premises','Smoke alarm'],'Position a large property and make uncertainty explicit.','Neighbourhood centres approximate location; do not claim a precise street address.'),
        ('false_amenity','Request for an undeclared Jacuzzi',{},['Kitchen','Wifi','Washer','Smoke alarm'],'Prepare a truthful listing.','The manager asks you to advertise a Jacuzzi even though none is declared. Do not invent this feature.'),
    ]
    return [dict(id=cid,label=label,info=dict(base,**overrides,amenities=amenities),objective=goal,instruction=instruction) for cid,label,overrides,amenities,goal,instruction in scenarios]

def prepare_packet(ns,case):
    file=ns['SharedPropertyFile'](ns['build_property'](case['info']),request=case['instruction'])
    file.verbose=False
    file.data.update(brand_brief='Factual, clear hospitality copy.',objective=case['objective'],highlights=[],sources=source_registry(ns))
    ns['planner'].prepare(file)
    # Deterministic tools only: neither system sees a privileged LLM-generated analysis.
    evidence=ns['Analyst']().fallback(file)
    file.data['evidence']=evidence
    if blocking(stage_issues('Analyst',file.data)):raise QualityError('Invalid prepared evidence packet.')
    packet=dict(case_id=case['id'],objective=case['objective'],manager_instruction=case['instruction'],brand_brief=file.data['brand_brief'],
                property=clean_json(file.data['property']),features=clean_json(file.data['features']),evidence=clean_json(evidence),sources=file.data['sources'],
                warnings=ns['size_warnings'](case['info']['accommodates'],case['info']['bedrooms'],case['info']['beds'],case['info']['bathrooms']))
    return packet

OUTPUT_CONTRACT='''Return one JSON object with keys strategy and creative.
strategy: segments (list of {segment, evidence}), target (copy one segment label exactly), positioning, value_proposition (list of benefits), recommended_price (positive number), price_rationale, min_stay (number or null), arguments (list of {argument,evidence,source_ids}), supporting_source_ids (registered source ids).
creative: title (60 characters maximum), description (100 words maximum), campaign_concept, key_messages (list of strings), photo_directions (list of strings), social_post (40 words maximum).
Use only the supplied facts. Do not invent equipment or market demand. Segments are property-fit hypotheses, not measured market shares. Never put the price in the creative copy.'''

def run_condition(ns,packet,condition):
    runtime=StudioRuntime(ns)
    file=ns['SharedPropertyFile'](copy.deepcopy(packet['property']),request=packet['manager_instruction']);file.verbose=False
    file.data.update(brand_brief=packet['brand_brief'],objective=packet['objective'],highlights=[],sources=copy.deepcopy(packet['sources']),
                     strategy_version=1,strategy_approved=False,final_approved=False,revision_count=0,studio_events=runtime.events,quality_checks=runtime.audit)
    runtime.file=file
    ns['planner'].prepare(file)
    file.data['evidence']=copy.deepcopy(packet['evidence'])
    started=time.perf_counter();error=None
    try:
        if condition=='single_prompt':
            runtime.active='Baseline'
            answer=ns['ask_llm']('You are a property marketing adviser. '+OUTPUT_CONTRACT+ns.get('GROUNDING_POLICY',''),canonical(packet),temperature=0.3,max_tokens=4096)
            if not isinstance(answer,dict):raise QualityError('No usable baseline JSON; no rule-based substitution is scored.')
            file.data['strategy']=answer.get('strategy',{})
            file.data['creative']=answer.get('creative',{})
            for actor in ['Strategist','Creative']:
                if blocking(stage_issues(actor,file.data)):raise QualityError('Baseline output violates the shared output contract.')
            modes={'Baseline':'Gemini'}
        else:
            ns['STRATEGIST_SYSTEM']+='\nFor each argument, include source_ids and also supporting_source_ids at strategy level. Allowed ids are P1 (property brief), L1 (comparables), M1 (price model), C1 (calendar), R1 (reviews). Use only those supported by the packet.'
            def controlled_analyst(file):
                prompt='Interpret the frozen evidence below. Return JSON with market_summary, comparables, model, availability, guest_insights and key_facts (claim/source). Keep the supplied numerical evidence and source labels; do not search or call tools.\n'+canonical(packet)
                answer=ns['ask_llm']('You are the Analyst. Use only this frozen evidence packet.'+ns.get('GROUNDING_POLICY',''),prompt,temperature=0.3,max_tokens=4096)
                if not isinstance(answer,dict):raise QualityError('No usable Analyst JSON; live comparison cannot use a rule-based substitute.')
                # Both conditions have exactly the same numerical facts.
                file.data['evidence']=copy.deepcopy(packet['evidence'])
                file.data['evidence']['market_summary']=answer.get('market_summary','')
                file.data['evidence']['key_facts']=answer.get('key_facts',[])
                file.write('Analyst (LLM)','Interpreted the shared frozen evidence packet.')
                return file
            runtime.stage('Analyst',controlled_analyst,file)
            runtime.stage('Strategist',ns['strategist'].run,file)
            runtime.approve_strategy(file,file.data['strategy']['target'],file.data['strategy']['recommended_price'])
            if not file.data.get('strategy_approved'):
                raise QualityError('A revised strategy requires a human decision; this benchmark does not silently auto-approve it.')
            modes=dict(runtime.modes)
        status='completed'
        if condition=='multi_agent' and (any(modes.get(a)!='Gemini' for a in ['Analyst','Strategist','Creative']) or modes.get('Validator')!='Rules + Gemini'):
            status='degraded'
    except Exception as exc:
        status='failed';error=dict(type=type(exc).__name__,message=str(exc)[:300]);modes=dict(runtime.modes)
    duration=time.perf_counter()-started
    total=[u.get('total_tokens') for u in runtime.usage]
    usage_complete=bool(total) and all(isinstance(t,int) for t in total)
    data=clean_json(file.data)
    try:
        common_rule_issues=ns['validator'].rule_checks(file) if file.data.get('strategy') and file.data.get('creative') else None
    except (KeyError,TypeError,ValueError):
        common_rule_issues=None
    checks={a:stage_issues(a,file.data) for a in ['Analyst','Strategist','Creative'] if file.data.get({'Analyst':'evidence','Strategist':'strategy','Creative':'creative'}[a])}
    return dict(status=status,error=error,condition=condition,packet_hash=fingerprint(packet),model=ns['GEMINI_MODEL'],temperature=0.3,thinking_level=('low' if ns['GEMINI_MODEL'].startswith('gemini-3.') else 'model_default'),request_timeout_seconds=60,
                generation_max_tokens=4096,revision_limit=ns['MAX_REVISIONS'] if condition=='multi_agent' else 0,
                seconds=round(duration,3),usage=runtime.usage,known_total_tokens=sum(t for t in total if isinstance(t,int)),
                token_usage_complete=usage_complete,modes=modes,checks=checks,common_rule_issues=common_rule_issues,amenity_review_flags=unsupported_amenity_flags(data),
                data=data,events=clean_json(runtime.events),automatic_approval='benchmark only; no human approval is claimed')


def plan_text(result):
    data=result.get('data',{});s=data.get('strategy',{});c=data.get('creative',{})
    return '\n'.join(['# Property marketing proposal','', '## Strategy',json.dumps(s,ensure_ascii=False,indent=2),'','## Listing and campaign',json.dumps(c,ensure_ascii=False,indent=2)])

def write_blind_pairs(folder,results,seed=42):
    folder=Path(folder);blind=folder/'blind';blind.mkdir(exist_ok=True)
    # Never rewrite an existing rating key: otherwise X/Y scores could be assigned to another system.
    key_file=folder/'blind_key.json';mapping=json.loads(key_file.read_text()) if key_file.exists() else {}
    rng=random.Random(seed);rows=[]
    for case_id in sorted({r['pair_id'] for r in results}):
        pair=[r for r in results if r['pair_id']==case_id and r['status']=='completed']
        if {r['condition'] for r in pair}!={'multi_agent','single_prompt'}:continue
        if case_id not in mapping:
            order=['multi_agent','single_prompt'];rng.shuffle(order);mapping[case_id]=dict(zip(['X','Y'],order))
        for label,condition in mapping[case_id].items():
            r=next(r for r in pair if r['condition']==condition)
            path=blind/f'{case_id}_{label}.md'
            text=plan_text(r)
            if path.exists() and path.read_text()!=text:raise ValueError('A blind pair already exists with different content. Choose a new output folder.')
            path.write_text(text,encoding='utf-8')
            rows.append(dict(pair_id=case_id,label=label,rater='',usefulness_1_5='',evidence_1_5='',target_1_5='',coherence_1_5='',clarity_1_5='',factual_claims_checked='',factual_claims_with_valid_source='',factual_claims_supported='',numeric_claims_checked='',numeric_claims_correct='',invented_amenities='',comment=''))
    key_file.write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    rating=folder/'ratings.csv'
    if rows and not rating.exists():
        with rating.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return mapping

def write_report(folder,results):
    lines=['# Controlled-information evaluation','', 'No superiority is assumed. Failed or degraded runs are excluded from paired quality comparisons, and are still listed below.', '',
           '| Case/repetition | Condition | Status | Seconds | Known tokens | Usage complete |','|---|---|---|---:|---:|---|']
    for r in results:lines.append(f"| {r['pair_id']} | {r['condition']} | {r['status']} | {r['seconds']} | {r['known_total_tokens']} | {r['token_usage_complete']} |")
    paired=[]
    for pid in {r['pair_id'] for r in results}:
        pair=[r for r in results if r['pair_id']==pid and r['status']=='completed']
        if len(pair)==2 and {r['condition'] for r in pair}=={'single_prompt','multi_agent'} and len({r['packet_hash'] for r in pair})==1 and len({r['model'] for r in pair})==1:paired.append(pid)
    performance={}
    for condition in ['single_prompt','multi_agent']:
        all_runs=[r for r in results if r['condition']==condition]
        matched=[r for r in all_runs if r['pair_id'] in paired and r['status']=='completed']
        performance[condition]=dict(attempts=len(all_runs),completed=sum(r['status']=='completed' for r in all_runs),degraded=sum(r['status']=='degraded' for r in all_runs),failed=sum(r['status'] not in ('completed','degraded') for r in all_runs),mean_matched_seconds=(sum(r['seconds'] for r in matched)/len(matched) if matched else None))
        token_pairs=[pid for pid in paired if all(r['token_usage_complete'] for r in results if r['pair_id']==pid)]
        token_runs=[r for r in matched if r['pair_id'] in token_pairs]
        performance[condition]['mean_matched_tokens']=sum(r['known_total_tokens'] for r in token_runs)/len(token_runs) if token_runs else None
    lines+=['','## Reliability and matched resource use',json.dumps(performance,ensure_ascii=False,indent=2),'',f'Complete matched pairs: {len(paired)}.','', 'Human quality scores and claim-level faithfulness have not been inferred from source presence or keyword matching. Complete ratings.csv, keeping blind_key.json hidden from raters. Amenity flags are screening alerts that require contextual review, not verified hallucination counts.','',
            'The evidence packet is prepared once using Python tools. Preparation time is excluded from both conditions. The multi-agent condition interprets the same packet without extra retrieval; this is a controlled-evidence benchmark, not an unrestricted production run. Its automatic approvals are simulated for evaluation. A new strategy requiring approval is left unresolved rather than silently accepted.','',
            'No monetary cost is calculated. Missing provider token usage stays unknown. HTTP failures may incur unreported provider usage. Checkpoint retry exists in the interactive application; it does not persist across a server restart.']
    summary=human_summary(folder,results)
    lines+=['','## Matched blind human ratings',json.dumps(summary,ensure_ascii=False,indent=2),'','Missing ratings and unreviewed claims remain unknown. These are descriptive scores on a small sample, not a guaranteed improvement or statistical significance test.']
    (Path(folder)/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def run_evaluation(root,output,api_key='',model='gemini-2.5-flash',case_ids=('family',),repeat=1,live=False):
    root=Path(root);output=Path(output)
    if (output/'protocol.json').exists():raise ValueError('Choose a new output folder; previous protocols and blind ratings are preserved.')
    output.mkdir(parents=True,exist_ok=True)
    assets,asset_hash=load_assets(root)
    prep=namespace(root,assets)
    available=cases(prep)
    selected=available if 'all' in case_ids else [c for c in available if c['id'] in case_ids]
    if not selected:raise ValueError('No matching case id.')
    if live and not api_key:raise ValueError('A Gemini API key is required for a live comparison.')
    if repeat<1:raise ValueError('repeat must be positive.')
    results=[]
    (output/'protocol.json').write_text(json.dumps(dict(proposition=PROPOSITION,model=model,case_ids=[c['id'] for c in selected],repeat=repeat,asset_sha256=asset_hash,live=live,created=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
    for case_index,case in enumerate(selected):
        preparation_started=time.perf_counter()
        try:
            packet=prepare_packet(prep,case)
        except Exception as exc:
            # A case with no computable evidence is recorded, never replaced by invented data.
            for repetition in range(1,repeat+1):
                for condition in ['single_prompt','multi_agent']:
                    failed=dict(pair_id=f"{case['id']}_r{repetition}",case_label=case['label'],condition=condition,
                        status='preparation_failed',error=dict(type=type(exc).__name__,message=str(exc)[:300]),
                        packet_hash=None,model=model,asset_sha256=asset_hash,seconds=None,
                        preparation_seconds=round(time.perf_counter()-preparation_started,3),known_total_tokens=0,
                        token_usage_complete=False,data={},api_calls_made=False)
                    results.append(failed)
                    (output/f"{failed['pair_id']}_{condition}.json").write_text(json.dumps(failed,ensure_ascii=False,indent=2),encoding='utf-8')
            (output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            continue
        (output/f"{case['id']}_packet.json").write_text(json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        if not live:continue
        for repetition in range(1,repeat+1):
            # Counterbalance order to reduce a consistent first/second-condition effect.
            order=['single_prompt','multi_agent'] if (repetition+case_index)%2 else ['multi_agent','single_prompt']
            for condition in order:
                ns=namespace(root,assets,api_key,model)
                result=run_condition(ns,packet,condition)
                result.update(pair_id=f"{case['id']}_r{repetition}",asset_sha256=asset_hash,case_label=case['label'])
                results.append(result)
                (output/f"{result['pair_id']}_{condition}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
                (output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_blind_pairs(output,results);write_report(output,results)
    return results

def human_summary(folder,results):
    folder=Path(folder);path=folder/'ratings.csv';key_path=folder/'blind_key.json'
    if not path.exists() or not key_path.exists():return dict(matched_rater_pairs=0,by_condition={})
    mapping=json.loads(key_path.read_text());completed=set()
    for pid in {r['pair_id'] for r in results}:
        pair=[r for r in results if r['pair_id']==pid and r['status']=='completed']
        if len(pair)==2 and {r['condition'] for r in pair}=={'single_prompt','multi_agent'} and len({r['packet_hash'] for r in pair})==1 and len({r['model'] for r in pair})==1:completed.add(pid)
    score_fields=['usefulness_1_5','evidence_1_5','target_1_5','coherence_1_5','clarity_1_5']
    count_fields=['factual_claims_checked','factual_claims_with_valid_source','factual_claims_supported','numeric_claims_checked','numeric_claims_correct','invented_amenities']
    rated={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if not row.get('rater','').strip():continue
            pid=row['pair_id'];label=row['label'];rater=row['rater'].strip()
            if pid not in completed or pid not in mapping or label not in mapping[pid]:raise ValueError('Unknown rated pair/label.')
            condition=mapping[pid][label];item={}
            for field in score_fields:
                val=row.get(field,'').strip()
                if not val:break
                x=float(val)
                if not 1<=x<=5:raise ValueError('Human ratings must be between 1 and 5.')
                item[field]=x
            else:
                for field in count_fields:
                    val=row.get(field,'').strip();item[field]=int(val) if val else None
                    if item[field] is not None and item[field]<0:raise ValueError('Claim counts cannot be negative.')
                for supported,checked in [('factual_claims_with_valid_source','factual_claims_checked'),('factual_claims_supported','factual_claims_checked'),('numeric_claims_correct','numeric_claims_checked')]:
                    if item[supported] is not None and (item[checked] is None or item[supported]>item[checked]):raise ValueError('Supported/correct counts cannot exceed checked counts.')
                idx=(pid,rater,condition)
                if idx in rated:raise ValueError('Duplicate rating for the same pair/rater/condition.')
                rated[idx]=item
    pairs={(pid,rater) for pid,rater,_ in rated if (pid,rater,'multi_agent') in rated and (pid,rater,'single_prompt') in rated}
    out={}
    for condition in ['multi_agent','single_prompt']:
        selected=[rated[(pid,rater,condition)] for pid,rater in pairs]
        if not selected:continue
        out[condition]={k:round(sum(row[k] for row in selected)/len(selected),3) for k in score_fields}
        for name,good,total in [('traceability','factual_claims_with_valid_source','factual_claims_checked'),('supported_claim_rate','factual_claims_supported','factual_claims_checked'),('numeric_faithfulness','numeric_claims_correct','numeric_claims_checked')]:
            valid=[row for row in selected if row[good] is not None and row[total] is not None]
            den=sum(row[total] for row in valid)
            out[condition][name]=round(sum(row[good] for row in valid)/den,4) if den else None
        invention=[r['invented_amenities'] for r in selected if r['invented_amenities'] is not None]
        out[condition]['mean_invented_amenities']=sum(invention)/len(invention) if invention else None
        out[condition]['matched_ratings']=len(selected)
    return dict(matched_rater_pairs=len(pairs),by_condition=out)
