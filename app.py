from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import threading
import joblib
import pandas as pd
import streamlit as st
from live_runtime import StudioRuntime
import studio_views as view

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Concierge Studio · Bordeaux',page_icon='✳',layout='wide',initial_sidebar_state='expanded')
view.html('<style>'+ (ROOT/'studio.css').read_text(encoding='utf-8')+'</style>')

@st.cache_resource
def load_assets():
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    blob=b''.join((ROOT/'assets'/name).read_bytes() for name in manifest['parts'])
    if hashlib.sha256(blob).hexdigest()!=manifest['sha256']: raise ValueError('Incomplete or modified assets export.')
    payload=joblib.load(BytesIO(blob))
    import shap
    payload['explainer']=shap.TreeExplainer(payload['tree_for_shap'])
    return payload,threading.RLock()

with st.sidebar:
    view.html('<div class="brand">concierge<em>✳</em><br>studio</div><div class="side-label">BORDEAUX / WORKSPACE</div>')
    st.write('')
    page=st.radio('Workspace',['Adviser studio','Price laboratory','Project & methods'],label_visibility='collapsed')
    st.divider()
    view.html('<div class="side-label">THE TEAM</div><div class="side-note">Planner · Analyst · Strategist<br>Creative · Validator</div>')

if not (ROOT/'assets/manifest.json').exists():
    view.hero()
    st.info('Ajoute le dossier assets déjà exporté à côté de app.py. Si tu as la première version sur GitHub, conserve son dossier assets : aucun nouvel export n’est nécessaire.')
    st.stop()
try:
    with st.spinner('Opening your market workspace…'): assets,model_lock=load_assets()
except Exception as exc:
    st.error('The trained models could not be loaded. Check assets and the library versions in requirements.txt.');st.exception(exc);st.stop()
try:
    api_key=st.secrets.get('GEMINI_API_KEY',os.environ.get('GEMINI_API_KEY',''))
    model_name=st.secrets.get('GEMINI_MODEL','gemini-2.5-flash')
except FileNotFoundError:
    api_key,model_name=os.environ.get('GEMINI_API_KEY',''),'gemini-2.5-flash'
config_hash=hashlib.sha256((str(api_key)+str(model_name)).encode()).hexdigest()
if 'studio_v2' not in st.session_state or st.session_state.get('studio_config')!=config_hash:
    ns=dict(assets,__name__='airbnb_runtime',_api_key=api_key,_model=model_name)
    exec(compile((ROOT/'engine_source.py').read_text(encoding='utf-8'),'engine_source.py','exec'),ns)
    st.session_state.studio_v2=StudioRuntime(ns)
    st.session_state.studio_config=config_hash
rt=st.session_state.studio_v2
ns=rt.ns
with st.sidebar:
    view.html('<div class="side-label">MARKET SNAPSHOT</div>')
    st.caption(f"{len(assets['data']):,} properties · Bordeaux\n\nInside Airbnb · September 2026")
    st.caption(f"Review method: {assets.get('nlp_method','unknown')}")
    st.divider()
    st.caption('Gemini configured' if api_key else 'Rule-based mode')
    if api_key: st.caption(f"Requested model: {model_name}")
    st.caption('Actual mode is shown for each completed agent. An API key does not guarantee a usable model response.')
    st.caption('Session memory only. Nothing is published to Airbnb.')

if page=='Project & methods':
    view.hero(rt.file)
    view.project_notes(assets)
    st.stop()

if page=='Price laboratory':
    view.section('Price laboratory','Change a scenario. Compare the predicted price immediately.')
    st.caption('Uses your trained model, with no Gemini request. Estimated associations are not guaranteed causal gains.')
    prior=rt.file.data['features'] if rt.file else {}
    cols=st.columns([1,1.3]);output=cols[1].empty()
    with cols[0]:
        with st.form('simulator'):
            names=ns['NEIGHBOURHOODS'];default=prior.get('neighbourhood',names[0])
            nb=st.selectbox('Neighbourhood',names,index=names.index(default) if default in names else 0)
            room=st.selectbox('Room type',ns['ROOM_TYPES'])
            a,b=st.columns(2)
            guests=a.number_input('Guests',1,30,int(prior.get('accommodates',4)))
            bedrooms=b.number_input('Bedrooms',0,20,int(prior.get('bedrooms',2)))
            bathrooms=st.number_input('Bathrooms',0.5,15.0,float(prior.get('n_bathrooms',1)),0.5)
            checks=st.multiselect('Equipment',[x[0] for x in ns['SIM_BINARIES']],default=[label for label,var in ns['SIM_BINARIES'] if prior.get(var)])
            safety=st.slider('Safety items',0,4,int(prior.get('n_safety',1)))
            amenities=st.slider('Number of amenities',0,100,min(100,int(prior.get('n_amenities',25))))
            rating=st.slider('Rating',0.0,5.0,4.7,0.1)
            new=st.checkbox('No rating yet',True);host=st.checkbox('Superhost')
            ref=st.number_input('Reference price (€), 0 for the first scenario',min_value=0.0,value=0.0)
            simulate=st.form_submit_button('Compare this scenario',type='primary')
    if simulate:
        with model_lock:
            st.session_state.sim_result=ns['simulate'](nb,room,guests,bedrooms,bathrooms,checks,safety,amenities,rating,new,host,ref or st.session_state.get('sim_reference'))
        if not st.session_state.get('sim_reference'):st.session_state.sim_reference=st.session_state.sim_result[3]
    if 'sim_result' in st.session_state:
        card,fig,market,reference=st.session_state.sim_result
        with cols[1]:
            view.html('<div class="studio-sim">'+card+'</div>');view.chart(fig,'sim-levers');view.chart(market,'sim-market')
    else:
        with output.container():view.html('<div class="info-card"><div class="eyebrow">YOUR SCENARIO</div><h4>What could change the price?</h4><p>Try a different neighbourhood, more capacity or additional amenities. The first scenario becomes your reference for subsequent comparisons.</p></div>')
    st.stop()

view.hero(rt.file)
view.section('Your specialist team','Real execution events · no simulated agent conversations')
pipeline=st.empty()
pipeline.markdown(view.agents_html(rt),unsafe_allow_html=True)

left,right=st.columns([1.65,1],gap='large')
with right:
    activity=st.empty()
    activity.markdown(view.activity_html(rt),unsafe_allow_html=True)
    current_context=st.empty()
    def render_context():
        file=rt.file
        if not file:
            body='Each agent reads the same property file. Evidence travels with the strategy and the campaign.'
        elif not file.data.get('strategy_approved'):
            body='Your next decision: review and approve the proposed strategy before the Creative agent continues.'
        elif not file.data.get('final_approved'):
            body='Your next decision: review the creative output and validator checks, then approve or request a revision.'
        else:body='This version is approved. Download the dossier or submit a new property brief.'
        current_context.markdown(f'<div class="info-card" style="margin-top:14px"><div class="eyebrow">SHARED CONTEXT / MANAGER CONTROL</div><p>{view.esc(body)}</p></div>',unsafe_allow_html=True)
    render_context()

def refresh(runtime):
    pipeline.markdown(view.agents_html(runtime),unsafe_allow_html=True)
    activity.markdown(view.activity_html(runtime),unsafe_allow_html=True)
    render_context()

names=ns['NEIGHBOURHOODS']
default_nb=next((n for n in names if n.startswith('Chartrons')),names[0])
def apply_example(family=True):
    st.session_state.update(p_nb=default_nb,p_room='Entire home/apt',p_guests=4 if family else 2,
        p_bedrooms=2 if family else 1,p_beds=2 if family else 1,p_bathrooms=1.0,p_surface=55 if family else 28,
        p_kitchen='Open kitchen',p_notes='Bright, recently renovated, close to the tram' if family else 'Quiet courtyard, compact city base',
        p_brand='Warm, straightforward hospitality. Focus on the amenities actually confirmed by the manager.',p_objective='Launch a new property')
    for n,choices in enumerate(ns['AMENITY_GROUPS'].values()):
        st.session_state[f'p_am_{n}']=[x for x in choices if x in (['Wifi','TV','Heating','Kitchen','Washer','Smoke alarm','Hot water','Bed linens'] if family else ['Wifi','Heating','Kitchen','Dedicated workspace','Smoke alarm'])]

with left:
    with st.expander('01 / Property & marketing brief',expanded=rt.file is None):
        st.caption('Start with an example, then edit the details to match the real property.')
        a,b=st.columns(2)
        if a.button('Family apartment',use_container_width=True):apply_example(True)
        if b.button('City break for two',use_container_width=True):apply_example(False)
        with st.form('property'):
            a,b=st.columns([1.2,1])
            nb=a.selectbox('Neighbourhood',names,key='p_nb')
            room=b.selectbox('Room type',ns['ROOM_TYPES'],key='p_room')
            a,b,c=st.columns(3)
            guests=a.number_input('Guests',1,30,4,key='p_guests')
            bedrooms=b.number_input('Bedrooms',0,20,2,key='p_bedrooms')
            beds=c.number_input('Beds',1,30,2,key='p_beds')
            a,b,c=st.columns(3)
            bathrooms=a.number_input('Bathrooms',0.5,15.0,1.0,0.5,key='p_bathrooms')
            surface=b.number_input('Surface (m²)',0,500,55,key='p_surface',help='Use 0 if unknown.')
            kitchen=c.selectbox('Kitchen',ns['KITCHEN_CHOICES'],index=1,key='p_kitchen')
            notes=st.text_area('Property highlights',placeholder='Confirmed details: renovated, near the tram, step-free access…',height=85,key='p_notes')
            with st.expander('Amenities & safety'):
                groups=[]
                for n,(label,choices) in enumerate(ns['AMENITY_GROUPS'].items()):
                    groups.append(st.multiselect(label,choices,key=f'p_am_{n}'))
            with st.expander('Brand context & business objective'):
                brand=st.text_area('Brand promise & tone',placeholder='Who are you as a concierge brand? What must the agents respect?',key='p_brand')
                objective=st.selectbox('Marketing objective',['Launch a new property','Reposition an existing property','Improve listing consistency'],key='p_objective')
            with st.expander('Stay dates (optional)'):
                a,b=st.columns(2)
                start=a.date_input('Check-in',value=None,key='p_start')
                end=b.date_input('Check-out',value=None,key='p_end')
                st.caption(f"Dataset calendar: {assets['cal']['date'].min().date()} to {assets['cal']['date'].max().date()}.")
            submitted=st.form_submit_button('Run the agents',type='primary',use_container_width=True)
    view.html('<div class="side-note">Only include confirmed features. The manager brief is treated as input, not independent market evidence.</div>')

if submitted:
    ds=start.isoformat() if start else None;de=end.isoformat() if end else None
    if bool(start)!=bool(end) or (ds and ds>=de):st.error('Select both dates, with check-out after check-in.')
    elif ds and (pd.Timestamp(ds)<assets['cal']['date'].min() or pd.Timestamp(de)>assets['cal']['date'].max()+pd.Timedelta(days=1)):
        st.error('The dates are outside the dataset calendar. Choose covered dates or clear both dates.')
    else:
        rt.on_update=refresh
        try:
            with model_lock:
                rt.analyse([nb,room,guests,bedrooms,beds,bathrooms,surface,kitchen,notes,ds,de,*groups],brand,objective)
            for key in ['target_edit_v2','price_edit_v2']:st.session_state.pop(key,None)
            refresh(rt)
        except Exception as exc:
            st.error('This run could not be completed. Review the failed stage in the activity log.');st.exception(exc)
        finally:rt.on_update=None

file=rt.file
if file is not None and file.data.get('strategy'):
    view.metrics(file)
    for w in file.data.get('warnings',[]):st.warning(w)
    pending=not file.data.get('strategy_approved')
    if pending:
        view.html('<div class="gate"><b>YOUR DECISION / APPROVE THE STRATEGY</b><p>Review the target, value proposition and price below. The Creative agent is paused until you approve this version.</p></div>')
        with st.form('strategy_approval'):
            a,b=st.columns([2,1])
            target=a.text_input('Target audience',str(file.data['strategy']['target']),key='target_edit_v2')
            price=b.number_input('Price per night (€)',min_value=1.0,value=float(file.data['strategy']['recommended_price']),key='price_edit_v2')
            approved=st.form_submit_button('Approve strategy & brief Creative',type='primary')
        if approved:
            rt.on_update=refresh
            try:
                with model_lock:rt.approve_strategy(file,target,price)
                for k in ['target_edit_v2','price_edit_v2']:st.session_state.pop(k,None)
                st.rerun()
            except Exception as exc:st.exception(exc)
            finally:rt.on_update=None
    tabs=st.tabs(['Market & sources','Marketing strategy','Creative studio','Shared context'])
    with model_lock:
        with tabs[0]:view.evidence(file,ns)
        with tabs[1]:view.strategy(file)
        with tabs[2]:view.creative(file)
        with tabs[3]:view.shared_context(file,ns,rt)
    if file.data.get('strategy_approved') and file.data.get('creative'):
        view.section('Manager review','Approve the result or send it back with clear feedback')
        if file.data.get('final_approved'):
            st.success('You approved this version. The listing and campaign are ready for export.')
        else:
            valid=file.data.get('validation',{}).get('status')=='VALIDATED'
            if st.button('Approve final listing & campaign',type='primary',disabled=not valid):
                rt.approve_final(file);st.rerun()
            if not valid:st.caption('Final approval unlocks when the validator’s outstanding issues are resolved.')
        with st.expander('Request a revision'):
            with st.form('revision'):
                actor=st.selectbox('Send feedback to',['Creative','Strategist'])
                feedback=st.text_area('What should change?',placeholder='Explain the issue or the marketing choice to revisit.')
                send=st.form_submit_button('Send feedback to the agent')
            if send:
                rt.on_update=refresh
                try:
                    with model_lock:rt.request_revision(file,actor,feedback)
                    for k in ['target_edit_v2','price_edit_v2']:st.session_state.pop(k,None)
                    st.rerun()
                except Exception as exc:st.exception(exc)
                finally:rt.on_update=None
    if file.data.get('final_approved'):
        report={k:v for k,v in file.data.items() if k!='property'}
        c=file.data['creative'];s=file.data['strategy']
        copy=f"{c.get('title','')}\n\n{c.get('description','')}\n\nCAMPAIGN\n{c.get('campaign_concept','')}\n\nSOCIAL POST\n{c.get('social_post','')}\n\nTARGET: {s.get('target')}\nNIGHTLY PRICE: €{s.get('recommended_price')}"
        a,b=st.columns(2)
        a.download_button('Download the approved dossier',json.dumps(report,ensure_ascii=False,indent=2,default=str),file_name='approved_property_dossier.json',mime='application/json',use_container_width=True)
        b.download_button('Download listing & campaign copy',copy,file_name='listing_and_campaign.txt',mime='text/plain',use_container_width=True)
else:
    view.section('A workflow with a clear handoff','Evidence stays connected to the final message')
    a,b,c=st.columns(3)
    for col,n,title,body in [(a,'01','Understand the market','Comparable properties, guest insights and a model-based price range.'),(b,'02','Choose a direction','Segmentation, a clear target and a proposition supported by the property.'),(c,'03','Create with control','A listing and campaign, checked against the evidence and approved by you.')]:
        with col:view.html(f'<div class="info-card"><div class="card-num">{n}</div><h4>{title}</h4><p>{body}</p></div>')
view.html('<div class="footnote">Inside Airbnb · Bordeaux · September 2026 · CC BY 4.0. Availability does not identify confirmed bookings. Model intervals and marketing recommendations require manager judgement. This prototype does not publish listings automatically.</div>')
