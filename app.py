from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import threading
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Bordeaux Airbnb Adviser', page_icon='🏠', layout='wide')
st.title('Bordeaux Airbnb Concierge Adviser')
st.caption('Five agents · Evidence-backed pricing · Manager approval')

@st.cache_resource
def load_assets():
    manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
    blob = b''.join((ROOT / 'assets' / name).read_bytes() for name in manifest['parts'])
    if hashlib.sha256(blob).hexdigest() != manifest['sha256']:
        raise ValueError('Export incomplet : un fichier assets est manquant ou modifié.')
    # Load only your own locally generated export, never a visitor upload.
    payload = joblib.load(BytesIO(blob))
    import shap
    payload['explainer'] = shap.TreeExplainer(payload['tree_for_shap'])
    return payload, threading.RLock()

if not (ROOT / 'assets/manifest.json').exists():
    st.info('Il reste à exporter les données et les modèles depuis ton notebook. Consulte GUIDE_FR.md, étape 2.')
    st.stop()
try:
    with st.spinner('Loading the trained models and market data…'):
        assets, model_lock = load_assets()
except Exception as exc:
    st.error('Impossible de charger l’export. Vérifie les fichiers assets et les versions dans requirements.txt.')
    st.exception(exc)
    st.stop()

try:
    api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))
    model_name = st.secrets.get('GEMINI_MODEL', 'gemini-2.5-flash')
except FileNotFoundError:
    api_key, model_name = os.environ.get('GEMINI_API_KEY', ''), 'gemini-2.5-flash'

if 'engine' not in st.session_state:
    # Separate function globals, CURRENT, agent state and memory for each browser session.
    ns = dict(assets, __name__='airbnb_runtime', _api_key=api_key, _model=model_name)
    exec(compile((ROOT / 'engine_source.py').read_text(encoding='utf-8'), 'engine_source.py', 'exec'), ns)
    st.session_state.engine = ns
ns = st.session_state.engine
st.caption('Gemini enabled; rule-based fallback if unavailable.' if api_key else 'Rule-based mode — add GEMINI_API_KEY in Streamlit Secrets to enable Gemini.')
st.caption('Reviews analysis: ' + str(assets.get('nlp_method', 'unknown')))

def html(value):
    st.markdown(value, unsafe_allow_html=True)

def render_result(file):
    html(ns['kpi_html'](file))
    tabs = st.tabs(['Market evidence', 'Strategy', 'Listing & validation', 'Agent conversation'])
    with tabs[0]:
        left, right = st.columns(2)
        with left: st.plotly_chart(ns['map_figure'](file), use_container_width=True)
        with right: st.plotly_chart(ns['price_figure'](file), use_container_width=True)
        left, right = st.columns(2)
        with left: st.plotly_chart(ns['drivers_figure'](file), use_container_width=True)
        with right: st.plotly_chart(ns['reviews_figure'](), use_container_width=True)
        html(ns['insights_html'](file))
    with tabs[1]:
        st.json(file.data['strategy'])
    with tabs[2]:
        html(ns['listing_html'](file, approved=st.session_state.get('approved', False)))
        if file.data.get('validation'): st.json(file.data['validation'])
    with tabs[3]: html(ns['feed_html'](file.data['log']))

st.markdown('<style>' + ns['CSS'] + '</style>', unsafe_allow_html=True)
main, simulator = st.tabs(['Property adviser', 'What-if simulator'])
with main:
    with st.form('property'):
        st.subheader('1. Describe the property')
        a, b, c = st.columns(3)
        nb = a.selectbox('Neighbourhood', ns['NEIGHBOURHOODS'])
        room = b.selectbox('Room type', ns['ROOM_TYPES'])
        guests = c.number_input('Guests', 1, 30, 4)
        a, b, c, d = st.columns(4)
        bedrooms = a.number_input('Bedrooms', 0, 20, 2)
        beds = b.number_input('Beds', 1, 30, 2)
        bathrooms = c.number_input('Bathrooms', 0.5, 15.0, 1.0, 0.5)
        surface = d.number_input('Surface (m²), 0 if unknown', 0, 500, 55)
        kitchen = st.selectbox('Kitchen', ns['KITCHEN_CHOICES'], index=1)
        notes = st.text_area('Highlights / property description', placeholder='Bright, renovated, near the tram…')
        a, b = st.columns(2)
        start = a.text_input('Check-in (YYYY-MM-DD, optional)')
        end = b.text_input('Check-out (YYYY-MM-DD, optional)')
        groups = []
        for label, choices in ns['AMENITY_GROUPS'].items():
            groups.append(st.multiselect(label, choices))
        submit = st.form_submit_button('Analyse the property', type='primary')
    if submit:
        ds, de = ns['norm_date'](start), ns['norm_date'](end)
        import pandas as pd
        invalid = bool(start or end) and (not ds or not de or ds >= de)
        if invalid:
            st.error('Enter both dates, with check-out after check-in.')
        elif ds and (pd.Timestamp(ds) < assets['cal']['date'].min() or pd.Timestamp(de) > assets['cal']['date'].max() + pd.Timedelta(days=1)):
            st.error('These dates are outside the dataset calendar. Leave dates empty or choose dates covered by the data.')
        else:
            try:
                with st.spinner('Planner → Analyst → Strategist…'), model_lock:
                    for result in ns['run_analysis'](nb, room, guests, bedrooms, beds, bathrooms, surface, kitchen, notes, ds, de, *groups):
                        if result[0] is not None: st.session_state.file = result[0]
                st.session_state.approved = False
                st.session_state.pop('target_edit', None)
                st.session_state.pop('price_edit', None)
            except Exception as exc:
                st.session_state.pop('file', None)
                st.error('Analysis failed. Check the data and property inputs.')
                st.exception(exc)
    file = st.session_state.get('file')
    if file is not None and file.data.get('strategy'):
        with model_lock: render_result(file)
        st.subheader('2. Approve the strategy and generate the listing')
        with st.form('strategy'):
            target = st.text_input('Target', value=str(file.data['strategy']['target']), key='target_edit')
            price = st.number_input('Price per night (€)', min_value=1.0, value=float(file.data['strategy']['recommended_price']), key='price_edit')
            generate = st.form_submit_button('Approve strategy & create listing')
        if generate:
            try:
                with st.spinner('Creative → Validator…'), model_lock:
                    for result in ns['run_creation'](file, target, price):
                        st.session_state.file = result[0]
                st.session_state.approved = False
                st.rerun()
            except Exception as exc: st.exception(exc)
        if file.data.get('creative'):
            st.subheader('3. Final manager approval')
            if file.data.get('validation', {}).get('status') != 'VALIDATED':
                st.warning('The validator found issues. Review the listing and strategy before approving.')
            if st.button('Approve final listing', disabled=st.session_state.get('approved', False)):
                file.data['approvals'].append(dict(step='final listing', decision='approved', by='manager'))
                ns['MEMORY'].save(file)
                st.session_state.approved = True
                st.rerun()
            if st.session_state.get('approved'):
                st.success('Approved by the manager.')
                st.download_button('Download property report (JSON)', json.dumps({k:v for k,v in file.data.items() if k != 'property'}, ensure_ascii=False, indent=2, default=str), file_name='property_report.json', mime='application/json')
with simulator:
    st.subheader('Compare a property scenario — no Gemini call')
    with st.form('simulator'):
        snb = st.selectbox('Neighbourhood', ns['NEIGHBOURHOODS'], key='snb')
        sr = st.selectbox('Room type', ns['ROOM_TYPES'], key='sr')
        a,b,c = st.columns(3)
        sg=a.number_input('Guests',1,30,4,key='sg')
        sb=b.number_input('Bedrooms',0,20,2,key='sb')
        sba=c.number_input('Bathrooms',0.5,15.0,1.0,0.5,key='sba')
        checks=st.multiselect('Equipment', [x[0] for x in ns['SIM_BINARIES']])
        safety=st.slider('Safety items',0,4,1)
        amenities=st.slider('Number of amenities',0,100,25)
        rating=st.slider('Rating',0.0,5.0,4.7,0.1)
        new=st.checkbox('No rating yet',value=True)
        host=st.checkbox('Superhost')
        simulate=st.form_submit_button('Simulate')
    if simulate:
        with model_lock:
            card, fig, market, reference=ns['simulate'](snb,sr,sg,sb,sba,checks,safety,amenities,rating,new,host,None)
        html(card)
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(market,use_container_width=True)
st.caption('Inside Airbnb, Bordeaux, September 2026 · CC BY 4.0. Unavailable calendar days are not confirmed bookings. Session memory resets when the session ends.')
