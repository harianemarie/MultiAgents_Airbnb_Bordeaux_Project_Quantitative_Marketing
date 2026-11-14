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
st.set_page_config(page_title='Airbnb Concierge Adviser, Bordeaux',page_icon='🏠',layout='wide',initial_sidebar_state='collapsed')
view.html('<style>'+ (ROOT/'studio.css').read_text(encoding='utf-8')+'</style>')

@st.cache_resource
def load_assets():
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    blob=b''.join((ROOT/'assets'/name).read_bytes() for name in manifest['parts'])
    if hashlib.sha256(blob).hexdigest()!=manifest['sha256']:raise ValueError('Incomplete or modified assets export.')
    payload=joblib.load(BytesIO(blob))
    import shap
    payload['explainer']=shap.TreeExplainer(payload['tree_for_shap'])
    return payload,threading.RLock()

if not (ROOT/'assets/manifest.json').exists():
    view.hero()
    st.info('Ajoute le dossier assets déjà exporté à côté de app.py. Conserve aussi ton requirements.txt actuel et tes secrets. Aucun nouvel entraînement n’est nécessaire.')
    st.stop()
try:
    with st.spinner('Loading your trained models and market data…'):assets,model_lock=load_assets()
except Exception as exc:
    st.error('Could not load the export. Check assets and the versions in requirements.txt.');st.exception(exc);st.stop()
try:
    api_key=st.secrets.get('GEMINI_API_KEY',os.environ.get('GEMINI_API_KEY',''))
    model_name=st.secrets.get('GEMINI_MODEL','gemini-2.5-flash')
except FileNotFoundError:
    api_key,model_name=os.environ.get('GEMINI_API_KEY',''),'gemini-2.5-flash'
config_hash=hashlib.sha256((str(api_key)+str(model_name)).encode()).hexdigest()
if 'studio_v3' not in st.session_state or st.session_state.get('studio_config')!=config_hash:
    ns=dict(assets,__name__='airbnb_runtime',_api_key=api_key,_model=model_name)
    exec(compile((ROOT/'engine_source.py').read_text(encoding='utf-8'),'engine_source.py','exec'),ns)
    st.session_state.studio_v3=StudioRuntime(ns)
    st.session_state.studio_config=config_hash
rt=st.session_state.studio_v3
ns=rt.ns
mode=f"Gemini configured · {model_name}" if ns.get('llm_client') is not None else 'Rule-based mode, Gemini off'
view.hero(rt.file,mode)
page=st.radio('Application',['Adviser studio','Price laboratory','Project & methods'],horizontal=True,
    format_func=lambda x:{'Adviser studio':'Agents: new property','Price laboratory':'Simulator','Project & methods':'Project & methods'}[x],label_visibility='collapsed')
if page=='Project & methods':
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

# Property inputs remain on the left; actions, agents and outputs remain on the right.
names=ns['NEIGHBOURHOODS']
default_nb=next((n for n in names if n.startswith('Chartrons')),names[0])

def apply_example(family=True):
    st.session_state.update(p_nb=default_nb,p_room='Entire home/apt',p_guests=4 if family else 2,
        p_bedrooms=2 if family else 1,p_beds=2 if family else 1,p_bathrooms=1.0,p_surface=55 if family else 28,
        p_kitchen='Open kitchen',p_notes='Bright, recently renovated, close to the tram' if family else 'Quiet courtyard, compact city base',
        p_brand='Warm, factual hospitality. Showcase confirmed property features.',p_objective='Launch a new property')
    checked=['Wifi','TV','Heating','Kitchen','Washer','Smoke alarm','Hot water','Bed linens'] if family else ['Wifi','Heating','Kitchen','Dedicated workspace','Smoke alarm']
    for n,choices in enumerate(ns['AMENITY_GROUPS'].values()):
        for j,item in enumerate(choices):st.session_state[f'p_cb_{n}_{j}']=item in checked

def tick_small_items():
    n=len(ns['AMENITY_GROUPS'])-1
    for j,_ in enumerate(list(ns['AMENITY_GROUPS'].values())[-1]):st.session_state[f'p_cb_{n}_{j}']=True

left,right=st.columns([0.41,0.59],gap='large')
with left:
    view.html('<div class="form-title">1. Describe the property</div>')
    a,b=st.columns(2)
    if a.button('Example: family flat',use_container_width=True):apply_example(True)
    if b.button('Example: studio',use_container_width=True):apply_example(False)
    with st.container(border=True):
        a,b=st.columns(2)
        nb=a.selectbox('Neighbourhood',names,key='p_nb')
        room=b.selectbox('Property type',ns['ROOM_TYPES'],key='p_room')
    with st.container(border=True):
        a,b=st.columns(2)
        start=a.date_input('Check-in (optional)',value=None,key='p_start')
        end=b.date_input('Check-out (optional)',value=None,key='p_end')
    with st.container(border=True):
        a,b=st.columns(2)
        guests=a.number_input('Guests',1,30,4,key='p_guests')
        bedrooms=b.number_input('Bedrooms',0,20,2,key='p_bedrooms')
        a,b=st.columns(2)
        beds=a.number_input('Beds',1,30,2,key='p_beds')
        bathrooms=b.number_input('Bathrooms',0.5,15.0,1.0,0.5,key='p_bathrooms')
    with st.container(border=True):
        a,b=st.columns(2)
        surface=a.number_input('Surface in m² (0 if unknown)',0,500,0,key='p_surface')
        kitchen=b.selectbox('Kitchen',ns['KITCHEN_CHOICES'],index=2,key='p_kitchen')
    with st.expander('Amenities: tick everything the property has',expanded=True):
        groups=[]
        for n,(label,choices) in enumerate(ns['AMENITY_GROUPS'].items()):
            with st.container(border=True):
                view.html(f'<span class="chip">{view.esc(label)}</span>')
                cells=st.columns(2)
                selected=[]
                for j,item in enumerate(choices):
                    with cells[j%2]:
                        if st.checkbox(ns['AMENITY_LABELS'].get(item,item),key=f'p_cb_{n}_{j}'):selected.append(item)
                groups.append(selected)
        st.button('Tick all everyday small items',on_click=tick_small_items,use_container_width=True)
    with st.container(border=True):
        notes=st.text_area('Highlights (optional, separated by commas)',placeholder='e.g. renovated, quiet courtyard, step-free access',height=95,key='p_notes')
    with st.expander('Brand context & marketing objective'):
        brand=st.text_area('Brand promise & tone',placeholder='Describe your concierge brand and the tone to respect.',key='p_brand')
        objective=st.selectbox('Marketing objective',['Launch a new property','Reposition an existing property','Improve listing consistency'],key='p_objective')
    st.caption('Tick confirmed amenities only. Calendar dates must be covered by the exported dataset.')

with right:
    view.html('<div class="form-title">2. Run the agents</div>')
    submitted=st.button('Run the agents',type='primary',use_container_width=True)
    pipeline=st.empty()
    live_note=st.empty()
    tabs=st.tabs(['Market evidence','Strategy and approval','Listing & campaign',"Agents’ conversation",'Shared context'])
    with tabs[0]:market_slot=st.empty()
    with tabs[1]:strategy_slot=st.empty()
    with tabs[2]:creative_slot=st.empty()
    with tabs[3]:activity=st.empty()
    with tabs[4]:shared_slot=st.empty()

market_render_id=[0]
def render_market():
    market_render_id[0]+=1
    with market_slot.container():
        if rt.file and rt.file.data.get('evidence'):
            if rt.file.data.get('strategy'):view.metrics(rt.file)
            view.evidence(rt.file,ns,key_prefix=f'workbench-{market_render_id[0]}-')
        else:view.market_placeholder()

def refresh(runtime):
    pipeline.markdown(view.agents_html(runtime),unsafe_allow_html=True)
    activity.markdown(view.activity_html(runtime),unsafe_allow_html=True)
    if runtime.events:
        event=runtime.events[-1]
        active=any(s=='running' for s in runtime.states.values())
        live_note.markdown(f'<div class="run-note"><i class="live-dot {"on" if active else ""}"></i><b>{view.esc(event["actor"])}</b> · {view.esc(event["message"])}</div>',unsafe_allow_html=True)
        if event['kind']=='complete' and event['actor'] in ('Analyst','Strategist'):
            render_market()
    else:
        live_note.markdown('<div class="run-note">Your agents are ready. Complete the property details, then run the analysis.</div>',unsafe_allow_html=True)

# The callback sends updates before and after real tool and agent operations.
if submitted:
    ds=start.isoformat() if start else None;de=end.isoformat() if end else None
    if bool(start)!=bool(end) or (ds and ds>=de):
        with right:st.error('Select both dates, with check-out after check-in.')
    elif ds and (pd.Timestamp(ds)<assets['cal']['date'].min() or pd.Timestamp(de)>assets['cal']['date'].max()+pd.Timedelta(days=1)):
        with right:st.error('These dates are outside the dataset calendar. Choose covered dates or clear both dates.')
    else:
        rt.on_update=refresh
        try:
            with model_lock:rt.analyse([nb,room,guests,bedrooms,beds,bathrooms,surface,kitchen,notes,ds,de,*groups],brand,objective)
            for key in ['target_edit_v3','price_edit_v3']:st.session_state.pop(key,None)
        except Exception as exc:
            with right:st.error('The run could not finish. Inspect the activity log.');st.exception(exc)
        finally:rt.on_update=None
refresh(rt)
with model_lock:render_market()
file=rt.file

with strategy_slot.container():
    if file is None or not file.data.get('strategy'):
        st.info('The proposed target, positioning and price will appear here after the analysis.')
    else:
        view.strategy(file)
        for warning in file.data.get('warnings',[]):st.warning(warning)
        if not file.data.get('strategy_approved'):
            view.html('<div class="gate"><b>YOUR TURN / APPROVE THE STRATEGY</b><p>Edit the target or price if necessary, then approve. Creative remains paused until your decision.</p></div>')
            with st.form('strategy_approval'):
                target=st.text_input('Target audience',str(file.data['strategy']['target']),key='target_edit_v3')
                price=st.number_input('Price per night (€)',min_value=1.0,value=float(file.data['strategy']['recommended_price']),key='price_edit_v3')
                approved=st.form_submit_button('Approve strategy & brief Creative',type='primary',use_container_width=True)
            if approved:
                rt.on_update=refresh
                try:
                    with model_lock:rt.approve_strategy(file,target,price)
                    for key in ['target_edit_v3','price_edit_v3']:st.session_state.pop(key,None)
                    st.rerun()
                except Exception as exc:st.exception(exc)
                finally:rt.on_update=None
        else:st.success('The manager approved this strategy. Review the Listing & campaign tab.')

with creative_slot.container():
    if file is None:st.info('Approve a strategy to generate the listing and campaign.')
    else:
        view.creative(file)
        if file.data.get('strategy_approved') and file.data.get('creative'):
            view.section('Final manager approval')
            valid=file.data.get('validation',{}).get('status')=='VALIDATED'
            if file.data.get('final_approved'):st.success('You approved this version. It is ready for export.')
            elif st.button('Approve final listing & campaign',type='primary',disabled=not valid,use_container_width=True):
                rt.approve_final(file);st.rerun()
            if not valid:st.caption('Resolve the validator’s outstanding issues before approving the final package.')
        if file.data.get('strategy'):
            with st.expander('Send feedback to an agent'):
                with st.form('revision'):
                    actor=st.selectbox('Send feedback to',['Creative','Strategist'])
                    feedback=st.text_area('What should change?')
                    send=st.form_submit_button('Send feedback to the agent')
                if send:
                    rt.on_update=refresh
                    try:
                        with model_lock:rt.request_revision(file,actor,feedback)
                        for key in ['target_edit_v3','price_edit_v3']:st.session_state.pop(key,None)
                        st.rerun()
                    except Exception as exc:st.exception(exc)
                    finally:rt.on_update=None
        if file.data.get('final_approved'):
            report={k:v for k,v in file.data.items() if k!='property'}
            c=file.data['creative'];s=file.data['strategy']
            text=f"{c.get('title','')}\n\n{c.get('description','')}\n\nCAMPAIGN\n{c.get('campaign_concept','')}\n\nSOCIAL POST\n{c.get('social_post','')}\n\nTARGET: {s.get('target')}\nNIGHTLY PRICE: €{s.get('recommended_price')}"
            st.download_button('Download the approved dossier',json.dumps(report,ensure_ascii=False,indent=2,default=str),file_name='approved_property_dossier.json',mime='application/json',use_container_width=True)
            st.download_button('Download listing & campaign copy',text,file_name='listing_and_campaign.txt',mime='text/plain',use_container_width=True)
with shared_slot.container():
    if file is None:st.info('The shared property file will appear once the Planner starts.')
    else:view.shared_context(file,ns,rt)
view.html('<div class="footnote">Inside Airbnb · Bordeaux · September 2026 · CC BY 4.0. Availability is not confirmed bookings. Actual Gemini/rule-based mode is shown for each agent. Session memory only. Nothing is published automatically.</div>')
