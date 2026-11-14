import html as _html
import re
import json
import streamlit as st

PALETTE = {'Planner':'#7772ac','Analyst':'#367c90','Strategist':'#cf8a3e','Creative':'#41896d','Validator':'#56636e'}
SHORT_ROLES = {'Planner':'Structure the property brief', 'Analyst':'Ground the decision in evidence', 'Strategist':'Choose a market position', 'Creative':'Build the listing & campaign', 'Validator':'Check consistency & sources'}

def esc(x): return _html.escape(str(x if x is not None else ''))
def md(x):
    s=esc(x)
    return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s).replace('\n','<br>')
def html(s): st.markdown(s,unsafe_allow_html=True)

def hero(file=None):
    place=esc(file.data['features'].get('neighbourhood')) if file else 'Bordeaux'
    html(f'''<div class="topbar"><div><b>CONCIERGE STUDIO</b> &nbsp; / &nbsp; PROPERTY MARKETING</div><span class="tag">BORDEAUX EDITION · M2 PROJECT 04</span></div>
    <div class="hero"><div><div class="eyebrow">FROM MARKET EVIDENCE TO A READY-TO-REVIEW CAMPAIGN</div>
    <h1>Your property.<br><em>One coherent story.</em></h1><p>A specialist team for every new listing. Watch the agents research, decide and create, then approve the choices that matter.</p><div class="hero-note">Five roles &nbsp; · &nbsp; Shared evidence &nbsp; · &nbsp; You make the final decision</div></div>
    <div class="hero-brief"><div class="eyebrow">THE WORKFLOW</div><h3>A new property.<br>A clear marketing direction.</h3><div class="brief-row"><b>For</b><span>Concierge & property managers</span></div><div class="brief-row"><b>Decision</b><span>Target, position, price and message</span></div><div class="brief-row"><b>Market</b><span>{place} · Inside Airbnb</span></div></div></div>''')

def agents_html(rt):
    labels={'waiting':'Queued','running':'Working','done':'Complete','approval':'Your approval','review':'Needs review','error':'Error'}
    out=['<div class="agents">']
    for n,(name,color) in enumerate(PALETTE.items(),1):
        state=rt.states[name];css={'running':'active','done':'done','approval':'blocked','review':'blocked','error':'blocked'}.get(state,'')
        timing=f"{rt.durations[name]:.1f}s · {rt.modes.get(name, '')}" if name in rt.durations else f'Step 0{n}'
        out.append(f'<div class="agent {css}" style="--agent:{color}"><div class="agent-top"><div class="agent-icon">{name[0]}</div><span class="agent-state">{labels[state]}</span></div><h4>{name}</h4><p>{SHORT_ROLES[name]}</p><small>{esc(timing)}</small></div>')
    return ''.join(out)+'</div>'

def activity_html(rt):
    active=any(s=='running' for s in rt.states.values())
    header=f'<div class="activity"><div class="activity-header"><span><i class="live-dot {"on" if active else ""}"></i> AGENT ACTIVITY</span><span>{"LIVE" if active else "EXECUTION LOG"}</span></div><div class="activity-body">'
    if not rt.events:
        return header+'<div class="empty-activity"><b>Your team is ready.</b>Submit the brief to see each agent work.<br>Tool calls, handoffs and approvals appear here.</div></div></div>'
    events=[]
    for e in reversed(rt.events[-60:]):
        actor=esc(e['actor']);kind=esc(e['kind'].replace('_',' '));msg=md(e['message'])
        source=f"<br><span class='evidence-chip'>{esc(e['source'])}</span>" if e.get('source') else ''
        events.append(f'<div class="event"><div class="event-time">+{e["elapsed"]:.1f}s</div><div><div class="event-who">{actor}<span class="event-type">{kind}</span></div><div class="event-msg">{msg}{source}</div></div></div>')
    return header+''.join(events)+'</div></div>'

def metrics(file):
    e=file.data['evidence'];s=file.data['strategy'];m=e['model'];c=e['comparables'];av=e.get('availability');rng=m['range_80pct']
    values=[('RECOMMENDED PRICE',f"€{float(s['recommended_price']):.0f}",'per night · manager can edit',True),
            ('MODEL ESTIMATE',f"€{float(m['predicted_price']):.0f}",f"80% interval: €{rng[0]:.0f}–€{rng[1]:.0f}",False),
            ('COMPARABLE PROPERTIES',str(c['n_comparables']),f"Median €{c['median']:.0f} · {c['radius_km']} km",False),
            ('COMPETITORS AVAILABLE',f"{av['share_comparables_free']:.0%}" if av else 'No dates', f"{av['n_comparables_free']} free for the stay" if av else 'Add dates to assess competition',False)]
    html('<div class="metric-grid">'+''.join(f'<div class="metric-card {"accent" if accent else ""}"><div class="metric-label">{esc(label)}</div><div class="metric-value">{esc(value)}</div><div class="metric-sub">{esc(sub)}</div></div>' for label,value,sub,accent in values)+'</div>')

def section(title,note=''):
    html(f'<div class="section-head"><h2>{esc(title)}</h2><p>{esc(note)}</p></div>')

def chart(fig,key):
    fig.update_layout(template='plotly_white',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Arial',size=11,color='#61746a'),title_font=dict(size=14,color='#263e33'))
    st.plotly_chart(fig,use_container_width=True,key=key,config={'displaylogo':False})

def evidence(file,ns):
    e=file.data['evidence']
    section('The market behind the recommendation','Source-grounded analysis')
    a,b=st.columns([1.15,1])
    with a: chart(ns['map_figure'](file),'market-map')
    with b: chart(ns['price_figure'](file),'market-price')
    html(f'<div class="info-card"><div class="eyebrow">ANALYST’S MARKET BRIEF</div><p>{md(e.get("market_summary", ""))}</p></div>')
    facts=e.get('key_facts',[])
    for f in facts:
        if isinstance(f,dict):
            html(f'<div class="info-card"><p>{md(f.get("claim"))}</p><span class="evidence-chip">{esc(f.get("source"))}</span></div>')
    a,b=st.columns(2)
    with a: chart(ns['drivers_figure'](file),'price-drivers')
    with b: chart(ns['reviews_figure'](),'review-themes')
    st.caption('SHAP describes the model, not a causal effect. Unavailable dates may be blocked by the host; they are not confirmed bookings.')
    section('Guest voice','Excerpts from comparable properties')
    review=ns['summarise_reviews']()
    for q in review.get('excerpts',[]):
        html(f'<div class="quote">“{esc(q.get("excerpt"))}”<small>Listing {esc(q.get("listing_id"))} · {esc(q.get("theme"))} · Inside Airbnb reviews</small></div>')
    with st.expander('Inspect the comparable listings'):
        comps=ns['CURRENT'].get('comparables')
        if comps is not None:
            cols=[c for c in ['id','price','accommodates','bedrooms','distance_km','score','available'] if c in comps]
            st.dataframe(comps.nlargest(20,'score')[cols],hide_index=True,use_container_width=True)


def strategy(file):
    s=file.data['strategy'];section('A position you can defend','Segmentation · Targeting · Positioning')
    st.caption('Candidate segments are hypotheses based on the property and evidence. This dataset does not identify the actual traveller type or prove segment demand.')
    html(f'<div class="positioning">{md(s.get("positioning"))}</div>')
    a,b=st.columns([1,1.2])
    with a:
        html(f'<div class="info-card"><div class="eyebrow">PRIMARY TARGET</div><h4>{esc(s.get("target"))}</h4><p>{md(s.get("price_rationale"))}</p></div>')
        for n,seg in enumerate(s.get('segments',[]),1):
            if isinstance(seg,dict): title=seg.get('segment',seg.get('name','Segment'));body=seg.get('evidence',seg.get('rationale',''))
            else:title=seg;body=''
            html(f'<div class="info-card"><div class="card-num">SEGMENT 0{n}</div><h4>{md(title)}</h4><p>{md(body)}</p></div>')
    with b:
        html('<div class="eyebrow">VALUE PROPOSITION</div>')
        vals=s.get('value_proposition',[])
        if isinstance(vals,str): vals=[vals]
        html(''.join(f'<span class="chip">{md(v)}</span>' for v in vals))
        st.write('')
        for arg in s.get('arguments',[]):
            if isinstance(arg,dict):
                html(f'<div class="info-card"><h4>{md(arg.get("argument"))}</h4><span class="evidence-chip">{esc(arg.get("evidence"))}</span></div>')
    if file.data.get('brand_brief'):
        html(f'<div class="info-card"><div class="eyebrow">MANAGER’S BRAND BRIEF</div><p>{md(file.data["brand_brief"])}</p><span class="evidence-chip">Manager input, not independently verified</span></div>')

def creative(file):
    c=file.data.get('creative',{});s=file.data['strategy']
    if not c:
        st.info('Approve the strategy to unlock the Creative agent.');return
    section('Your listing & campaign','Generated from the approved strategy')
    a,b=st.columns([1.2,1])
    with a:
        html(f'<div class="listing-visual"><span>BORDEAUX · {esc(s.get("target"))}</span></div><div class="listing-copy"><div class="eyebrow">LISTING PREVIEW</div><h2>{esc(c.get("title"))}</h2><p>{md(c.get("description"))}</p><div class="listing-price">€{float(s["recommended_price"]):.0f} <small>/ night</small></div></div>')
        st.caption('Decorative visual only. No property photograph has been generated or verified.')
    with b:
        html(f'<div class="info-card"><div class="eyebrow">CAMPAIGN CONCEPT</div><h4>{md(c.get("campaign_concept"))}</h4></div>')
        html('<div class="info-card"><div class="eyebrow">KEY MESSAGES</div>'+''.join(f'<p>{n:02d} &nbsp; {md(m)}</p>' for n,m in enumerate(c.get('key_messages',[]),1))+'</div>')
        html(f'<div class="info-card"><div class="eyebrow">SOCIAL POST</div><p>{md(c.get("social_post"))}</p></div>')
    section('Photo direction','A shot list for the manager or photographer')
    columns=st.columns(2)
    for n,item in enumerate(c.get('photo_directions',[]),1):
        with columns[(n-1)%2]: html(f'<div class="info-card"><div class="card-num">SHOT 0{n}</div><p>{md(item)}</p></div>')
    validation=file.data.get('validation',{})
    if validation.get('status')=='VALIDATED': st.success('Validator: coherence checks passed. Final manager approval is still required.' if not file.data.get('final_approved') else 'Validator checks passed. The manager approved this version.')
    else:
        st.warning('The validator found items to revise.')
        for i in validation.get('issues',[]): st.write(f"**{i.get('agent')}** · {i.get('issue')}")


def shared_context(file,ns,rt):
    section('One shared property file','Inspect what each agent received and produced')
    st.caption('These are inputs, tool results, outputs and handoffs. They do not expose a model’s private reasoning.')
    maps={'Planner':('features','Structured property profile'),'Analyst':('evidence','Sourced market evidence'),'Strategist':('strategy','Marketing choices'),'Creative':('creative','Execution package'),'Validator':('validation','Coherence checks')}
    for name,(key,label) in maps.items():
        with st.expander(name+' · '+label):
            cfg=ns['AGENT_CONFIG'][name]
            st.write('**Input:** '+cfg['input'])
            st.write('**Tools:** '+', '.join(cfg['tools']))
            st.caption(f"Temperature {cfg['llm']['temperature']} · token limit {cfg['llm']['max_tokens']} · actual mode: {rt.modes.get(name, 'Not run')}")
            st.json(json.loads(json.dumps(file.data.get(key,{}),default=str)))
    with st.expander('Manager decisions & revision history'):
        st.json(file.data['approvals'])
        st.json(file.data.get('revision_history',[]))
    st.download_button('Download execution trace',json.dumps(rt.events,ensure_ascii=False,indent=2,default=str),file_name='agent_execution_trace.json',mime='application/json',key='trace-export')


def project_notes(assets):
    section('A focused workflow, a defensible product','Project 4 · Multi-Agent Digital Marketing Adviser')
    html('<div class="info-card"><div class="eyebrow">JOB TO BE DONE</div><h4>Turn a newly onboarded property into a consistent marketing brief and listing.</h4><p>The recurring handoff is from market research to positioning, then listing copy. This prototype keeps the evidence with the decision and asks the manager to approve the strategy before producing creative work.</p></div>')
    st.table({'Role':['User','Potential buyer','Beneficiaries'],'Definition':['Concierge/property manager onboarding a new property','Concierge business owner or agency (commercial hypothesis)','Manager and property owner; guests receive a clearer listing']})
    st.subheader('How the app addresses the brief')
    st.dataframe([
        {'Project 4 requirement':'Evidence & market context','In this version':'Comparables, price range, guest-review themes and source labels'},
        {'Project 4 requirement':'Segmentation, targeting, positioning','In this version':'Readable strategy, value proposition and supporting arguments'},
        {'Project 4 requirement':'Creative execution','In this version':'Listing, campaign concept, key messages, social post and shot list'},
        {'Project 4 requirement':'Agent-to-agent coordination','In this version':'Real handoff events, shared property file and validator feedback'},
        {'Project 4 requirement':'Human approval points','In this version':'Strategy approval, renewed approval after strategy revisions, final approval'},
        {'Project 4 requirement':'Grounding & validation','In this version':'Source labels, Python checks, optional LLM review, visible issues'},
    ],hide_index=True,use_container_width=True)
    st.subheader('Methods from the first course')
    nlp=str(assets.get('nlp_method','unknown'))
    st.table([
        {'Course concept':'Pattern matching (slide 11)','Application':'Rule-based fallback when Gemini is unavailable; limitations displayed'},
        {'Course concept':'Sentiment & embeddings (slides 12, 30–32)','Application':f'Review themes exported from the notebook. Actual export method: {nlp}'},
        {'Course concept':'Supervised prediction (slide 10)','Application':'Trained model and quantile price interval exported without retraining'},
        {'Course concept':'Generation & summarisation (slide 33)','Application':'Gemini agents when a usable response is returned; explicit rules otherwise'},
        {'Course concept':'Visual analytics (slides 16, 35–41)','Application':'Photo directions are delivered. CLIP image inference is disabled in this lightweight deployment'},
    ])
    st.caption('The course surveys methods; this prototype selects those useful for this workflow. It does not claim to implement every method in the slides.')
    st.subheader('Alternatives & differentiation')
    st.table([
        {'Alternative category':'General-purpose copilot','Comparison to test':'Can produce text quickly; benchmark against the same brief and verify sources and consistency.'},
        {'Alternative category':'Pricing/market analytics software','Comparison to test':'Benchmark pricing quality separately from strategy-to-creative consistency.'},
        {'Alternative category':'Agency or internal workflow','Comparison to test':'Measure manual research time, revisions and unsupported claims on the same properties.'},
    ])
    st.info('Product hypothesis: a traceable connection from each market claim to an approved marketing choice and a coherent creative output. These are comparison questions, not measured competitor claims.')
    st.subheader('Commercial hypotheses to validate')
    a,b=st.columns(2)
    with a:
        st.markdown('**Business model:** a subscription per concierge agency, with usage limits to control API costs.\n\n**Go-to-market:** pilot with a small set of Bordeaux concierge managers; onboard real properties and collect structured feedback.\n\n**Pricing:** not validated. Interview managers and measure willingness to pay before proposing a tariff.')
    with b:
        st.markdown('**SWOT draft**\n\n- Strength: visible coordination and manager control.\n- Weakness: one city and one data snapshot.\n- Opportunity: repeated property onboarding.\n- Threat: stale data, API limits and inaccurate claims.')
    st.subheader('What still needs evidence')
    st.markdown('Compare against a manual workflow and a single copilot using the same held-out properties. Measure completion time, unsupported-claim rate, reviewer consistency scores and manager usefulness. The current run time is instrumented; no time savings or conversion uplift has been demonstrated. Technical test results remain in the notebook and must be reported with their original limitations.')
    st.info('Course submission (slide 4): the report must cover the problem, data exploration, solution and summary. Include the original notebook or reproducible script, the data and the presentation. This deployment package does not replace those research deliverables.')
    st.caption('Sources: Project Topics 2026–2027, common brief p. 2 and topic 4 p. 6; Big data in marketing 2026, slides 4, 10–16 and 27–41 as indicated above.')
