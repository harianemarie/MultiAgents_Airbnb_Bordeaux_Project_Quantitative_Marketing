"""Small, deterministic contracts. Checks do not replace a human factual review."""
import math
import re
import unicodedata

class QualityError(ValueError):
    pass

def normalise(value):
    text=unicodedata.normalize('NFKD',str(value)).encode('ascii','ignore').decode().casefold()
    return re.sub(r'[^a-z0-9]+',' ',text).strip()

def finite_number(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)

def issue(check,message,severity='blocking'):
    return dict(check=check,message=message,severity=severity)

def response_issues(actor,value):
    if not isinstance(value,dict):return [issue('json_object','The response must be a JSON object.')]
    text_fields={
        'Analyst':['market_summary'],
        'Strategist':['target','positioning','price_rationale'],
        'Creative':['title','description','campaign_concept','social_post'],
    }.get(actor,[])
    problems=[issue('required_text',f'Missing or empty field: {key}') for key in text_fields if not isinstance(value.get(key),str) or not value[key].strip()]
    if actor=='Analyst':
        for key in ['comparables','model']:
            if not isinstance(value.get(key),dict):problems.append(issue('required_object',f'{key} must be an object.'))
        facts=value.get('key_facts')
        if not isinstance(facts,list) or not facts:problems.append(issue('required_sources','key_facts must contain sourced claims.'))
        elif any(not isinstance(f,dict) or not isinstance(f.get('claim'),str) or not f['claim'].strip() or not isinstance(f.get('source'),str) or not f['source'].strip() for f in facts):
            problems.append(issue('claim_format','Every key fact needs a claim and a source.'))
    if actor=='Strategist':
        price=value.get('recommended_price')
        if not finite_number(price) or price<=0:problems.append(issue('positive_price','recommended_price must be a finite positive number.'))
        segments=value.get('segments')
        if not isinstance(segments,list) or not segments or any(not isinstance(s,dict) or not isinstance(s.get('segment'),str) or not s['segment'].strip() for s in segments):
            problems.append(issue('segments','segments must contain named candidate segments.'))
        elif normalise(value.get('target')) not in {normalise(s['segment']) for s in segments}:
            problems.append(issue('target_membership','Copy the target label exactly from one of the candidate segments.'))
        if not isinstance(value.get('arguments'),list) or any(not isinstance(a,dict) or not isinstance(a.get('argument'),str) or not isinstance(a.get('evidence'),str) for a in value.get('arguments',[])):
            problems.append(issue('arguments','arguments must be a list of argument/evidence objects.'))
        benefits=value.get('value_proposition')
        if not isinstance(benefits,list) or not benefits or not all(isinstance(v,str) and v.strip() for v in benefits):
            problems.append(issue('value_proposition','Provide a nonempty list of benefits.'))
    if actor=='Creative':
        for key in ['key_messages','photo_directions']:
            val=value.get(key)
            if not isinstance(val,list) or not val or not all(isinstance(v,str) and v.strip() for v in val):
                problems.append(issue('creative_items',f'{key} must be a nonempty list of text items.'))
        if len(value.get('title',''))>60:problems.append(issue('title_length','Listing title exceeds 60 characters.','warning'))
    if actor=='Validator':
        if not isinstance(value.get('issues'),list) or any(not isinstance(i,dict) or i.get('agent') not in ('Strategist','Creative') or not isinstance(i.get('issue'),str) for i in value.get('issues',[])):
            problems.append(issue('validator_format','issues must identify Strategist or Creative and a textual issue.'))
    return problems

def source_registry(ns):
    return [
        dict(id='P1',kind='manager',reference='Manager property brief'),
        dict(id='L1',kind='data',reference='Inside Airbnb listings, Bordeaux, 17 Sept 2026'),
        dict(id='M1',kind='model',reference=f"{ns['best_name']}, chosen by 5-fold cross-validation; quantile gradient boosting for the range"),
        dict(id='C1',kind='data',reference='Inside Airbnb calendar (free / not free per day)'),
        dict(id='R1',kind='data',reference='Inside Airbnb reviews of the comparables'),
    ]

def source_issues(data):
    registry=data.get('sources',[])
    known={normalise(s[k]) for s in registry for k in ['id','reference']}
    return [issue('unknown_source',f"Unknown source on key fact {n}: {f.get('source')}") for n,f in enumerate(data.get('evidence',{}).get('key_facts',[]),1) if isinstance(f,dict) and normalise(f.get('source','')) not in known]

def reference_id_issues(value,registry,path='output'):
    known={s['id'] for s in registry}
    problems=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('source_ids','supporting_source_ids'):
                if not isinstance(item,list) or any(not isinstance(v,str) or v not in known for v in item):
                    problems.append(issue('unknown_source_id',f'{path}.{key} contains an unknown source id or wrong type.'))
            elif key=='source_id':
                if not isinstance(item,str) or item not in known:problems.append(issue('unknown_source_id',f'{path}.source_id is unknown.'))
            else:problems+=reference_id_issues(item,registry,path+'.'+key)
    elif isinstance(value,list):
        for i,item in enumerate(value):problems+=reference_id_issues(item,registry,f'{path}[{i}]')
    return problems

def stage_issues(actor,data):
    key={'Analyst':'evidence','Strategist':'strategy','Creative':'creative','Validator':'validation'}.get(actor)
    problems=response_issues(actor,data.get(key)) if key else []
    if key:
        problems+=reference_id_issues(data.get(key),data.get('sources',[]))
    if actor=='Analyst' and not problems:
        problems+=source_issues(data)
        comparables=data['evidence']['comparables']
        for field in ['weighted_average_top20','p10','p25','median','p75','p90']:
            if not finite_number(comparables.get(field)) or comparables[field]<=0:problems.append(issue('comparable_price',f'Invalid comparable statistic: {field}'))
        if not finite_number(comparables.get('n_comparables')) or comparables['n_comparables']<1:problems.append(issue('empty_comparables','At least one comparable is required.'))
        model=data['evidence']['model'];rng=model.get('range_80pct',[])
        if not finite_number(model.get('predicted_price')) or model['predicted_price']<=0:
            problems.append(issue('model_price','The model prediction must be a positive finite number.'))
        if not isinstance(rng,list) or len(rng)!=2 or not all(finite_number(x) and x>0 for x in rng) or rng[0]>rng[1]:
            problems.append(issue('model_interval','The price interval is missing, non-finite or reversed.'))
    if actor=='Strategist' and not problems:
        s=data['strategy'];c=data['evidence']['comparables'];m=data['evidence']['model']
        low=min(c['p10'],m['range_80pct'][0]);high=max(c['p90'],m['range_80pct'][1])
        if not low<=s['recommended_price']<=high:
            problems.append(issue('price_range','The proposed price is outside the evidence range. Review and justify it.','warning'))
    return problems

def blocking(problems):return [p for p in problems if p['severity']=='blocking']

# This is a conservative screening tool, not a complete hallucination detector.
def unsupported_amenity_flags(data):
    f=data.get('features',{});c=data.get('creative',{})
    text=' '.join(str(c.get(k,'')) for k in ['title','description','campaign_concept','social_post'])+' '+' '.join(c.get('key_messages',[]))
    checks={'wifi':r'\bwi[ -]?fi\b','workspace':r'\b(?:dedicated workspace|work desk)\b','parking':r'\b(?:free parking|private parking|garage)\b','air_conditioning':r'\b(?:air conditioning|air-conditioned)\b','washer':r'\b(?:washing machine|washer)\b'}
    amenities=' '.join(f.get('amenity_list',[])).casefold()
    checks.update(jacuzzi=r'\b(?:jacuzzi|hot tub)\b',pool=r'\b(?:swimming pool|private pool)\b')
    flags=[]
    for key,pattern in checks.items():
        present=bool(f.get(key)) if key not in ('jacuzzi','pool') else bool(re.search(pattern,amenities))
        if present:continue
        for match in re.finditer(pattern,text,re.I):
            prefix=text[max(0,match.start()-65):match.start()]
            if re.search(r'\b(?:no|without|not|unavailable|not confirmed)\b[^.!?]{0,45}$',prefix,re.I):continue
            flags.append(dict(amenity=key,quote=text[max(0,match.start()-30):match.end()+30],status='needs_human_review'))
            break
    return flags
