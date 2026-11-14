import os, json, re, unicodedata, html, inspect, tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
USE_LLM = True                    # True once GEMINI_API_KEY is stored in Colab's Secrets panel
GEMINI_MODEL = "gemini-3.8-flash"  # check the current model name in Google AI Studio

BORDEAUX_CENTRE = (44.8425, -0.5745)
RADII_KM = [1.0, 2.0, 3.0]
MIN_COMPARABLES = 10
N_TOP_COMPARABLES = 20
SIMILARITY_WEIGHTS = {'location': 0.4, 'size': 0.2, 'amenities': 0.2, 'availability': 0.2}
MAX_QUOTE_NIGHTS = 27
IQR_FACTOR = 1.5
ROBUST_Z = 3.5
HORIZON_DAYS = 180
COMPETITION_THRESHOLDS = (0.3, 0.7)
COMPETITION_ADJUSTMENT = (1.10, 0.95)
MAX_REVISIONS = 2
USE_IMAGE_ANALYSIS = False

# Hugging Face Space: the same notebook runs as an app (section 16.1); FAST_MODE skips the long grid searches
IN_SPACE = bool(os.environ.get('SPACE_ID'))
FAST_MODE = IN_SPACE
USE_LLM = bool(_api_key)
USE_IMAGE_ANALYSIS = False
GEMINI_MODEL = _model
SAFETY = ['smoke alarm', 'carbon monoxide alarm', 'fire extinguisher', 'first aid kit']
def parse_amenities(text):
    try:
        return [a.lower() for a in json.loads(text)]
    except Exception:
        return []

def count_items(amenities, items):
    return sum(any(i in a for a in amenities) for i in items)

def has_amenity(amenities, word):
    return int(any(word in a for a in amenities))

def kitchen_type(description, amenities):
    # Keywords stay in French and English, since descriptions are written in both languages
    d = str(description).lower()
    if any(w in d for w in ['cuisine séparée', 'cuisine separee', 'cuisine indépendante', 'cuisine independante',
                            'separate kitchen', 'closed kitchen', 'cuisine fermée', 'cuisine fermee']):
        return 'separate'
    if any(w in d for w in ['kitchenette', 'cuisine ouverte', 'cuisine américaine', 'cuisine americaine',
                            'open kitchen', 'open-plan kitchen', 'coin cuisine']):
        return 'open'
    if has_amenity(amenities, 'kitchen'):
        return 'kitchen (type unknown)'
    return 'no kitchen'

def n_bathrooms(text):
    # "1.5 baths" -> 1.5 ; "Half-bath" -> 0.5
    t = str(text).lower()
    if 'half' in t and not re.search(r'\d', t):
        return 0.5
    m = re.search(r'(\d+(\.\d+)?)', t)
    return float(m.group(1)) if m else np.nan

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))
SURFACE_PATTERN = '(\\d{2,3}(?:[.,]\\d)?)\\s?(?:m²|m2|sqm|sq m|square met|mètres carrés|metres carres|m\\b)'
def extract_surface(text):
    m = re.search(SURFACE_PATTERN, str(text).lower())
    if not m:
        return np.nan
    value = float(m.group(1).replace(',', '.'))
    return value if 9 <= value <= 500 else np.nan

def iqr_outlier(values, groups, factor=IQR_FACTOR):
    q1 = values.groupby(groups).transform(lambda s: s.quantile(0.25))
    q3 = values.groupby(groups).transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return (values < q1 - factor * iqr) | (values > q3 + factor * iqr)
def availability(ids, start, end):
    # For each listing: 1 if it is free EVERY day from start (included) to end (excluded), otherwise 0
    period = cal[(cal['date'] >= pd.Timestamp(start)) & (cal['date'] < pd.Timestamp(end)) & cal['listing_id'].isin(ids)]
    free = period.groupby('listing_id')['free'].min()
    return free.reindex(ids).fillna(0).astype(int)

def minimum_stay(ids, start, end):
    # Minimum stay required by each listing on the first day of the period
    day = cal[(cal['date'] == pd.Timestamp(start)) & cal['listing_id'].isin(ids)]
    return day.set_index('listing_id')['minimum_nights'].reindex(ids)
def price_drivers(x_row, top=5):
    # Main contributions to the predicted price, in % (log scale → percentage)
    if explainer is None:
        return []
    values = explainer.shap_values(x_row)[0]
    s = pd.Series(values, index=x_row.columns)
    s = s[s.abs().sort_values(ascending=False).index][:top]
    labels = {c: c.replace('nb_', 'neighbourhood: ').replace('type_', 'room type: ') for c in s.index}
    return [{'feature': labels[c], 'value': float(x_row.iloc[0][c]), 'effect_pct': round(float(np.exp(v) - 1) * 100, 1)}
            for c, v in s.items()]
IMAGE_LABELS = {
    'separate kitchen': 'a photo of a separate closed kitchen',
    'open kitchen': 'a photo of an open kitchen in a living room',
    'balcony or terrace': 'a photo of a balcony or terrace',
    'living room': 'a photo of a living room',
    'bedroom': 'a photo of a bedroom',
    'fire extinguisher': 'a photo of a fire extinguisher',
}
_clip = None

def analyse_image(url, threshold=0.25):
    global _clip
    if not USE_IMAGE_ANALYSIS or not isinstance(url, str):
        return []
    try:
        import requests, torch
        from io import BytesIO
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        if _clip is None:
            _clip = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                     CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
        model, processor = _clip
        image = Image.open(BytesIO(requests.get(url, timeout=15).content)).convert("RGB")
        inputs = processor(text=list(IMAGE_LABELS.values()), images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=1)[0].numpy()
        return [name for name, p in zip(IMAGE_LABELS, probs) if p >= threshold]
    except Exception as e:
        print("Image analysis failed:", e)
        return []
llm_client = None
if USE_LLM:
    # !pip install -q google-genai     (uncomment in Colab if needed)
    from google import genai
    from google.genai import types
    api_key = _api_key
    if api_key:
        llm_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=20_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        print("Gemini ready:", GEMINI_MODEL)
    else:
        print("No GEMINI_API_KEY found: agents will use their rule-based fallbacks.")
else:
    print("USE_LLM = False: agents will use their rule-based fallbacks.")


def extract_json(text):
    # Gemini sometimes wraps JSON in ``` fences or adds a sentence: keep the outermost {...}
    if not text:
        return None
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def ask_llm(system, prompt, tools=None, temperature=0.3):
    # One call to Gemini. With tools, Gemini can call our Python functions before answering.
    if llm_client is None:
        return None
    try:
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            tools=tools or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=4) if tools else None,
        )
        response = llm_client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
        return extract_json(response.text)
    except Exception as e:
        print("Gemini call failed:", e)
        return None
# Several Gemini models, tried in order: the free tier has a small DAILY quota per model (e.g. 20 requests),
# and a model can be temporarily overloaded (error 503). When one model fails, the next one is used.
GEMINI_MODELS = list(dict.fromkeys([GEMINI_MODEL, 'gemini-2.5-flash']))
EXHAUSTED = set()          # models whose daily quota is used up in this session
ACTIVE_MODEL = GEMINI_MODEL
LLM_CALLS = 0

def ask_llm(system, prompt, tools=None, temperature=0.3, max_tokens=None):
    global ACTIVE_MODEL, LLM_CALLS
    if llm_client is None:
        return None
    config = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, max_output_tokens=max_tokens, tools=tools or None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=4) if tools else None)
    for model in [m for m in dict.fromkeys(GEMINI_MODELS) if m not in EXHAUSTED]:
        emit = globals().get('_network_event')
        if emit:
            emit('llm_attempt', model, 'Contacting this model (20 s per HTTP request).')
        try:
            response = llm_client.models.generate_content(model=model, contents=prompt, config=config)
            usage=getattr(response,'usage_metadata',None)
            record_usage=globals().get('_usage_event')
            if record_usage:
                record_usage(dict(model=model,prompt_tokens=getattr(usage,'prompt_token_count',None),
                                  output_tokens=getattr(usage,'candidates_token_count',None),
                                  total_tokens=getattr(usage,'total_token_count',None),
                                  thinking_tokens=getattr(usage,'thoughts_token_count',None)))
            ACTIVE_MODEL, LLM_CALLS = model, LLM_CALLS + 1
            return extract_json(response.text)
        except Exception as e:
            msg = str(e)
            if emit:
                emit('llm_error', model, 'Request failed or timed out; the agent will use a fallback if needed.')
            if '429' in msg or 'RESOURCE_EXHAUSTED' in msg:
                EXHAUSTED.add(model)
                print(f"Daily quota reached for {model} → trying the next model")
            elif any(code in msg for code in ['503', 'UNAVAILABLE', '404', 'NOT_FOUND']):
                print(f"{model} unavailable right now → trying the next model")
            else:
                print("Gemini call failed:", msg[:200])
                return None
    print("No Gemini model available: the agent uses its rule-based fallback.")
    return None

print("Models that will be tried:", [m for m in dict.fromkeys(GEMINI_MODELS)])
# Composition of each agent: LLM + memory + tools, with its own LLM parameters.
# temperature: low = factual and stable, high = more creative (and more risk of hallucination)
# max_tokens:  maximum length of the answer (Gemini "thinking" models also count their reasoning in it)
AGENT_CONFIG = {
    'Planner': {
        'role': 'Orchestrator: turns the manager request into a property profile and runs the agents in order',
        'llm': {'temperature': 0.0, 'max_tokens': 1024},
        'memory': 'writes the profile in the shared file; recalls similar past cases from the long-term memory',
        'tools': ['parse_request (Gemini: text → JSON)', 'features_of (data extractor)', 'analyse_image (photo analyser)',
                  'MEMORY.recall (past cases)'],
        'input': 'manager form or description, dates',
        'output': 'property profile (size, amenities, safety, kitchen, distance) + similar past cases',
    },
    'Analyst': {
        'role': 'Market analyst: gathers the evidence, every number comes from a tool',
        'llm': {'temperature': 0.1, 'max_tokens': 4096},
        'memory': 'reads the profile; writes the evidence brief',
        'tools': ['get_property_profile', 'find_comparables (similarity score)', 'predict_price_range (ML model + SHAP)',
                  'check_availability (calendar)', 'summarise_reviews (NLP on reviews)'],
        'input': 'property profile + dates',
        'output': 'evidence brief: comparables, model price and 80% range, availability, guest insights, sourced facts',
    },
    'Strategist': {
        'role': 'Marketing strategist: segmentation, targeting, positioning and price',
        'llm': {'temperature': 0.3, 'max_tokens': 2048},
        'memory': 'reads profile, evidence, past cases and Validator feedback; writes the strategy',
        'tools': ['evidence brief (no direct data access)', 'rule-based fallback'],
        'input': 'property profile + evidence brief (+ Validator feedback)',
        'output': 'target, positioning, price and rationale, minimum stay, arguments with evidence',
    },
    'Creative': {
        'role': 'Creative lead: turns the approved strategy into a listing and a campaign',
        'llm': {'temperature': 0.7, 'max_tokens': 2048},
        'memory': 'reads the approved strategy and the amenities; writes the creative output',
        'tools': ['approved strategy', 'amenity list', 'rule-based fallback'],
        'input': 'approved strategy + property amenities (+ Validator feedback)',
        'output': 'title, description, campaign concept, key messages, photo directions, social post',
    },
    'Validator': {
        'role': 'Quality control: checks that every price and claim is backed by the data',
        'llm': {'temperature': 0.0, 'max_tokens': 1024},
        'memory': 'reads every section; writes the verdict and sends issues back',
        'tools': ['rule checks (price range, target, amenities, no price in text)', 'Gemini consistency review'],
        'input': 'profile + evidence + strategy + creative output',
        'output': 'VALIDATED or TO REVIEW, with the issues sent back to the Strategist or the Creative agent',
    },
}

def llm_params(agent):
    return dict(temperature=AGENT_CONFIG[agent]['llm']['temperature'], max_tokens=AGENT_CONFIG[agent]['llm']['max_tokens'])

# Common writing rules added to every system prompt: short, readable outputs
STYLE = ("\nWriting rules: be concise. Each text field: at most 2 short sentences (35 words). "
         "Use Markdown inside text fields (**bold** for key figures). No emojis.")
STYLE_CREATIVE = ("\nWriting rules: plain, concise English, no emojis, no Markdown symbols in the title. "
                  "Description: at most 80 words.")


class AgentMemory:
    # Long-term memory: every approved analysis is saved in a JSON file and recalled for similar properties
    def __init__(self, path='agent_memory.json'):
        self.path = path
        self.cases = json.load(open(path)) if os.path.exists(path) else []

    def save(self, file):
        f, s, v = file.data['features'], file.data['strategy'], file.data.get('validation', {})
        if not s:
            return
        self.cases.append({'neighbourhood': f.get('neighbourhood'), 'room_type': f.get('room_type'),
                           'guests': f.get('accommodates'), 'bedrooms': f.get('bedrooms'), 'target': s.get('target'),
                           'price': s.get('recommended_price'), 'status': v.get('status'), 'dates': file.data.get('dates')})
        json.dump(self.cases, open(self.path, 'w'), default=str, indent=1)

    def recall(self, f, n=3):
        similar = [c for c in self.cases if c['neighbourhood'] == f.get('neighbourhood') and c['room_type'] == f.get('room_type')
                   and abs((c['guests'] or 0) - f.get('accommodates', 0)) <= 1]
        return similar[-n:]

MEMORY = None


def agent_card(agent):
    # Markdown card shown each time an agent starts: what goes in, with which tools, what comes out
    c = AGENT_CONFIG[agent]
    return (f"**{agent}**: {c['role']}  \n"
            f"*LLM:* Gemini, temperature {c['llm']['temperature']}, max {c['llm']['max_tokens']} tokens · *Memory:* {c['memory']}  \n"
            f"*Input:* {c['input']}  \n*Tools:* {', '.join(c['tools'])}  \n*Expected output:* {c['output']}")

for name in AGENT_CONFIG:
    print(f"{name:<11} T={AGENT_CONFIG[name]['llm']['temperature']:<4} max_tokens={AGENT_CONFIG[name]['llm']['max_tokens']:<5} "
          f"input: {AGENT_CONFIG[name]['input']}  →  output: {AGENT_CONFIG[name]['output']}")
class SharedPropertyFile:
    def __init__(self, property_data, dates=None, request=None):
        self.data = {'request': request, 'property': property_data, 'dates': dates,
                     'features': {}, 'evidence': {}, 'strategy': {}, 'creative': {}, 'validation': {},
                     'feedback': {}, 'approvals': [], 'memory': [], 'log': [], 'brand_brief': '', 'objective': ''}
        self.verbose = True

    def write(self, agent, message):
        self.data['log'].append(f"[{agent}] {message}")
        if self.verbose:
            print(f"[{agent}] {message}")

    def start(self, agent):
        # Announce the agent: what goes in, with which tools, what comes out
        c = AGENT_CONFIG[agent]
        self.data['log'].append(f"[{agent}] starts. Input: {c['input']}. Expected output: {c['output']}")
        if self.verbose:
            try:
                from IPython.display import display, Markdown
                display(Markdown(agent_card(agent)))
            except ImportError:
                print(agent_card(agent))

    def compact(self, *keys):
        # JSON summary of some sections, to give to an LLM
        return json.dumps({k: self.data[k] for k in keys}, ensure_ascii=False, default=str)

base = data[data['id'].isin(dev_ids)].copy()
CURRENT = {}          # the property being analysed, set by the Planner

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


def get_property_profile() -> dict:
    """Return the structured features of the property being analysed (size, amenities, location, safety, kitchen)."""
    f = CURRENT['features']
    return {k: v for k, v in f.items() if k != 'amenity_list'}


def find_comparables(radius_km: float = 1.0) -> dict:
    """Find listings of the same room type and similar capacity within radius_km of the property,
    rank them by similarity (location 0.4, size 0.2, amenities 0.2, availability 0.2) and summarise their prices."""
    f, dates = CURRENT['features'], CURRENT.get('dates')
    b = base[base['id'] != f['id']].copy()
    b['distance_km'] = haversine_km(f['latitude'], f['longitude'], b['latitude'], b['longitude'])
    cand = b[(b['room_type'] == f['room_type']) & (b['accommodates'].sub(f['accommodates']).abs() <= 2)]
    radii = [r for r in RADII_KM if r >= radius_km] or [radius_km]
    for radius in radii:
        comps = cand[cand['distance_km'] <= radius].copy()
        if len(comps) >= MIN_COMPARABLES:
            break
    if len(comps) < MIN_COMPARABLES:
        comps = cand.nsmallest(MIN_COMPARABLES, 'distance_km').copy()
        radius = round(float(comps['distance_km'].max()), 2)
    comps['available'] = availability(comps['id'].tolist(), *dates).values if dates else 1
    bedrooms = f['bedrooms'] if pd.notna(f['bedrooms']) else 1
    sim_loc = (1 - comps['distance_km'] / max(radius, 0.1)).clip(lower=0)
    sim_size = (1 - (comps['accommodates'].sub(f['accommodates']).abs() / 4
                     + comps['bedrooms'].sub(bedrooms).abs().clip(upper=2) / 4)).clip(lower=0)
    sim_am = comps['amenity_list'].apply(lambda a: jaccard(a, f['amenity_list']))
    W = SIMILARITY_WEIGHTS
    comps['score'] = W['location'] * sim_loc + W['size'] * sim_size + W['amenities'] * sim_am + W['availability'] * comps['available']
    top = comps.nlargest(N_TOP_COMPARABLES, 'score')
    CURRENT['comparables'] = comps
    return {
        'n_comparables': int(len(comps)), 'radius_km': float(radius),
        'weighted_average_top20': round(float(np.average(top['price'], weights=top['score'])), 1),
        'p10': round(float(comps['price'].quantile(.1)), 1), 'p25': round(float(comps['price'].quantile(.25)), 1),
        'median': round(float(comps['price'].median()), 1), 'p75': round(float(comps['price'].quantile(.75)), 1),
        'p90': round(float(comps['price'].quantile(.9)), 1),
        'property_better_equipped_than_share': round(float((comps['n_amenities'] < f['n_amenities']).mean()), 2),
        'top5': top.head(5)[['id', 'price', 'accommodates', 'bedrooms', 'review_scores_rating']]
                .assign(distance_km=top.head(5)['distance_km'].round(2)).to_dict('records'),
        'source': 'Inside Airbnb listings, Bordeaux, 17 Sept 2026',
    }


def predict_price_range() -> dict:
    """Predict the property's price with the validated model, an 80% price range (quantile models),
    and the main drivers of the prediction (SHAP, effect in %)."""
    x = CURRENT['x_row']
    return {'predicted_price': round(float(np.exp(best_model.predict(x))[0]), 1),
            'range_80pct': [round(float(np.exp(q_low.predict(x))[0]), 1), round(float(np.exp(q_high.predict(x))[0]), 1)],
            'drivers': price_drivers(x),
            'source': f'{best_name}, chosen by 5-fold cross-validation; quantile gradient boosting for the range'}


def check_availability() -> dict:
    """For the requested dates: share of comparables still free (competitive pressure) and their median minimum stay."""
    dates = CURRENT.get('dates')
    if not dates or 'comparables' not in CURRENT:
        return {'dates': None, 'note': 'no dates requested, or comparables not searched yet'}
    comps = CURRENT['comparables']
    ms = minimum_stay(comps['id'].tolist(), *dates)
    return {'dates': list(dates), 'share_comparables_free': round(float(comps['available'].mean()), 2),
            'n_comparables_free': int(comps['available'].sum()),
            'median_min_stay_nights': float(ms.median()) if ms.notna().any() else None,
            'source': 'Inside Airbnb calendar (free / not free per day)'}


def summarise_reviews() -> dict:
    """What guests mention in reviews of nearby comparables, compared with the Bordeaux average,
    with a few short review excerpts as evidence."""
    if 'comparables' not in CURRENT:
        return {'note': 'search comparables first'}
    comps = CURRENT['comparables']
    cols = ['reviews_' + t for t in THEMES]
    nearby = comps[cols].mean()
    themes = {c.replace('reviews_', ''): {'nearby_pct': round(float(nearby[c]) * 100, 1),
                                         'bordeaux_pct': round(float(REVIEW_AVERAGES[c]) * 100, 1)} for c in cols}
    above = sorted(themes, key=lambda t: themes[t]['nearby_pct'] - themes[t]['bordeaux_pct'], reverse=True)[:2]
    excerpts = []
    for theme in above:
        r = reviews[reviews['listing_id'].isin(comps['id']) & (reviews[theme] == 1)]
        for _, row in r.head(2).iterrows():
            excerpts.append({'listing_id': int(row['listing_id']), 'theme': theme,
                             'excerpt': str(row['comments'])[:140]})
    return {'themes': themes, 'themes_above_average': above, 'excerpts': excerpts,
            'source': 'Inside Airbnb reviews of the comparables'}


ANALYST_TOOLS = [get_property_profile, find_comparables, predict_price_range, check_availability, summarise_reviews]

NEIGHBOURHOODS = sorted(data['neighbourhood_cleansed'].unique())
NB_CENTRES = data.groupby('neighbourhood_cleansed')[['latitude', 'longitude']].median()
AMENITY_VOCAB = ['Wifi', 'Kitchen', 'Washer', 'Dishwasher', 'TV', 'Air conditioning', 'Heating',
                 'Free parking on premises', 'Dedicated workspace', 'Smoke alarm', 'Carbon monoxide alarm',
                 'Fire extinguisher', 'First aid kit', 'Elevator', 'Balcony', 'Coffee maker', 'Hair dryer',
                 'Iron', 'Hot water', 'Bed linens', 'Microwave', 'Refrigerator', 'Oven', 'Crib']


def parse_request(text):
    system = ("You are an inspector specialised in real-estate data extraction: you turn a property manager's description "
              "into structured data for an Airbnb pricing tool. Answer in JSON only.")
    prompt = f"""Extract the property as JSON with keys:
room_type (one of "Entire home/apt", "Private room", "Shared room", "Hotel room"), accommodates (int), bedrooms (int),
beds (int), bathrooms (number), surface_m2 (number or null), neighbourhood (exactly one of: {NEIGHBOURHOODS}),
amenities (list, only from: {AMENITY_VOCAB}), checkin and checkout (YYYY-MM-DD or null; current year 2026),
description (one factual sentence). Do not invent amenities that are not mentioned.

Description: {text}"""
    return ask_llm(system, prompt, **llm_params('Planner'))


def build_property(info):
    nb = info.get('neighbourhood')
    centre = NB_CENTRES.loc[nb] if nb in NB_CENTRES.index else pd.Series({'latitude': BORDEAUX_CENTRE[0], 'longitude': BORDEAUX_CENTRE[1]})
    surface = f" {info['surface_m2']} m²." if info.get('surface_m2') else ''
    return {'id': None, 'price': np.nan, 'room_type': info.get('room_type', 'Entire home/apt'),
            'accommodates': int(info.get('accommodates', 2)), 'bedrooms': float(info.get('bedrooms', 1)),
            'beds': float(info.get('beds', 1)), 'bathrooms_text': f"{info.get('bathrooms', 1)} baths",
            'latitude': float(centre['latitude']), 'longitude': float(centre['longitude']),
            'neighbourhood_cleansed': nb, 'amenities': json.dumps(info.get('amenities', [])),
            'description': info.get('description', '') + surface, 'picture_url': None}


def features_of(p):
    am = parse_amenities(p.get('amenities', '[]'))
    kitchen = kitchen_type(p.get('description', ''), am)
    if kitchen.startswith('kitchen') and 'separate kitchen' in analyse_image(p.get('picture_url')):
        kitchen = 'separate'
    return {'id': p.get('id'), 'room_type': p['room_type'], 'accommodates': int(p['accommodates']),
            'bedrooms': float(p.get('bedrooms') if pd.notna(p.get('bedrooms')) else 1), 'beds': float(p.get('beds') or 1),
            'n_bathrooms': n_bathrooms(p.get('bathrooms_text')), 'surface_m2': extract_surface(p.get('description')),
            'latitude': float(p['latitude']), 'longitude': float(p['longitude']), 'neighbourhood': p.get('neighbourhood_cleansed'),
            'amenity_list': am, 'n_amenities': len(am), 'n_safety': count_items(am, SAFETY),
            'safety_items': [s for s in SAFETY if any(s in a for a in am)],
            'parking': has_amenity(am, 'parking'), 'wifi': has_amenity(am, 'wifi'), 'workspace': has_amenity(am, 'workspace'),
            'air_conditioning': has_amenity(am, 'air conditioning'), 'washer': has_amenity(am, 'washer'),
            'kitchen': kitchen, 'separate_kitchen': int(kitchen == 'separate'),
            'dist_centre_km': round(float(haversine_km(p['latitude'], p['longitude'], *BORDEAUX_CENTRE)), 2)}


def x_row_of(p, f, index_label=None):
    if index_label is not None and index_label in X.index:
        return X.loc[[index_label]]
    row = pd.DataFrame([{c: 0.0 for c in X.columns}])
    values = {k: f[k] for k in ['accommodates', 'bedrooms', 'beds', 'n_bathrooms', 'n_amenities', 'n_safety', 'parking',
                                'air_conditioning', 'washer', 'workspace', 'separate_kitchen', 'dist_centre_km']}
    values.update({'superhost': 0, 'rating': data['rating'].median(), 'no_rating': 1})
    for k, v in values.items():
        row[k] = float(v if pd.notna(v) else 0)
    if f"type_{f['room_type']}" in row.columns:
        row[f"type_{f['room_type']}"] = 1.0
    nb_col = f"nb_{f['neighbourhood']}"
    if nb_col in row.columns:
        row[nb_col] = 1.0
    elif 'nb_Other neighbourhoods' in row.columns:
        row['nb_Other neighbourhoods'] = 1.0
    return row


def auto_approval(file, step):
    file.data['approvals'].append({'step': step, 'decision': 'approved', 'by': 'simulated manager'})
    file.write('Manager', f"{step}: approved (simulated)")
    return True


class Planner:
    name = "Planner"

    def prepare(self, file, index_label=None):
        file.start(self.name)
        p = file.data['property']
        f = features_of(p)
        file.data['memory'] = MEMORY.recall(f)
        file.data['features'] = {k: v for k, v in f.items()}
        CURRENT.clear()
        CURRENT.update({'features': f, 'dates': file.data['dates'], 'x_row': x_row_of(p, f, index_label)})
        file.write(self.name, f"Profile ready: {f['room_type']}, {f['accommodates']} guests, {f['n_amenities']} amenities, "
                              f"{f['n_safety']} safety items, kitchen: {f['kitchen']}, {f['dist_centre_km']} km from centre"
                              + (f", dates {file.data['dates'][0]} → {file.data['dates'][1]}" if file.data['dates'] else '')
                              + f" | memory: {len(file.data['memory'])} similar past case(s)")
        return file

    def run_until_strategy(self, file):
        file = analyst.run(file)
        file = strategist.run(file)
        return file

    def run_from_creative(self, file):
        file = creative.run(file)
        for round_ in range(MAX_REVISIONS + 1):
            file = validator.run(file)
            issues = file.data['validation']['issues']
            if not issues or round_ == MAX_REVISIONS:
                break
            agents_to_redo = {i['agent'] for i in issues}
            file.data['feedback'] = {a: [i['issue'] for i in issues if i['agent'] == a] for a in agents_to_redo}
            file.write(self.name, f"Validator sent work back to {sorted(agents_to_redo)} (revision {round_ + 1})")
            if 'Strategist' in agents_to_redo:
                file = strategist.run(file)
            file = creative.run(file)
        return file

    def run(self, file, approve=auto_approval, index_label=None):
        file = self.prepare(file, index_label)
        file = self.run_until_strategy(file)
        approve(file, 'target, positioning and price')
        file = self.run_from_creative(file)
        approve(file, 'final listing and campaign')
        MEMORY.save(file)
        return file

import re, unicodedata

def _norm(s):
    # lower case, no accents: "Chambre privée" -> "chambre privee"
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()

NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'un': 1, 'une': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10}
NUM = r'(\d+|' + '|'.join(NUMBER_WORDS) + r')'

def _to_int(word):
    return int(word) if word.isdigit() else NUMBER_WORDS[word]

def _find_number(patterns, text):
    for p in patterns:
        m = re.search(p, text)
        if m:
            return _to_int(m.group(1))
    return None

AMENITY_KEYWORDS = {
    'Wifi': ['wifi', 'wi-fi', 'internet'], 'Kitchen': ['kitchen', 'cuisine', 'kitchenette'],
    'Washer': ['washing machine', 'washer', 'lave-linge', 'lave linge', 'machine a laver'],
    'Dishwasher': ['dishwasher', 'lave-vaisselle', 'lave vaisselle'], 'TV': ['tv', 'television', 'tele'],
    'Air conditioning': ['air conditioning', 'air-conditioning', 'ac', 'a/c', 'clim', 'climatise', 'climatisation'],
    'Heating': ['heating', 'chauffage'], 'Free parking on premises': ['parking', 'garage'],
    'Dedicated workspace': ['workspace', 'desk', 'bureau', 'teletravail'],
    'Smoke alarm': ['smoke alarm', 'smoke detector', 'detecteur de fumee'],
    'Carbon monoxide alarm': ['carbon monoxide', 'monoxyde'], 'Fire extinguisher': ['extinguisher', 'extincteur'],
    'First aid kit': ['first aid', 'trousse de secours'], 'Elevator': ['elevator', 'lift', 'ascenseur'],
    'Balcony': ['balcony', 'balcon', 'terrace', 'terrasse'], 'Coffee maker': ['coffee', 'cafetiere', 'machine a cafe'],
    'Hair dryer': ['hair dryer', 'seche-cheveux', 'seche cheveux'], 'Iron': ['iron', 'fer a repasser'],
    'Microwave': ['microwave', 'micro-ondes', 'micro ondes'], 'Refrigerator': ['fridge', 'refrigerator', 'refrigerateur', 'frigo'],
    'Oven': ['oven'], 'Crib': ['crib', 'baby cot', 'lit bebe', 'lit parapluie'],
}

def parse_request_rules(text):
    t = _norm(text)

    # 1. Room type
    if re.search(r'private room|chambre privee|chambre chez', t):
        room_type = 'Private room'
    elif re.search(r'shared room|chambre partagee', t):
        room_type = 'Shared room'
    else:
        room_type = 'Entire home/apt'

    # 2. Numbers: guests, bedrooms, bathrooms
    guests = _find_number([NUM + r'\s*(?:people|persons|guests|personnes|voyageurs|pers\b)',
                           r'\bfor\s+' + NUM + r'\b', r'\bpour\s+' + NUM + r'\b'], t) or 2
    bedrooms = _find_number([NUM + r'[\s-]*(?:bedrooms?|chambres?)\b'], t)
    m = re.search(r'\bt(\d)\b', t)                       # French "T3" = 2 bedrooms
    if bedrooms is None and m:
        bedrooms = max(int(m.group(1)) - 1, 1)
    bedrooms = bedrooms or (1 if 'studio' in t else max(1, guests // 2))
    bathrooms = _find_number([NUM + r'\s*(?:bathrooms?|salles? de bains?)'], t) or 1

    # 3. Neighbourhood: look for a part of the official name in the text
    best, best_len = None, 0
    for nb in NEIGHBOURHOODS:
        for part in re.split(r'\s-\s|\(|\)', nb):
            part = _norm(part).strip()
            if len(part) >= 4 and part in t and len(part) > best_len:
                best, best_len = nb, len(part)

    # 4. Amenities (whole words only, to avoid false matches)
    amenities = [label for label, words in AMENITY_KEYWORDS.items()
                 if label in AMENITY_VOCAB and any(re.search(r'\b' + re.escape(w) + r'\b', t) for w in words)]

    # 5. Dates
    found = re.findall(r'\d{4}-\d{2}-\d{2}', text)
    checkin, checkout = (found[0], found[1]) if len(found) >= 2 else (None, None)
    if checkin is None and re.search(r'christmas|noel', t):
        checkin, checkout = '2026-12-19', '2026-12-26'
    elif checkin is None and re.search(r'new year|nouvel an', t):
        checkin, checkout = '2026-12-29', '2027-01-02'

    return {'room_type': room_type, 'accommodates': guests, 'bedrooms': bedrooms, 'beds': max(bedrooms, guests // 2),
            'bathrooms': bathrooms, 'surface_m2': extract_surface(text) if pd.notna(extract_surface(text)) else None,
            'neighbourhood': best, 'amenities': amenities, 'checkin': checkin, 'checkout': checkout,
            'description': str(text).split('.')[0][:200]}

# Keep the Gemini version, and use the rules as a fallback
if '_parse_request_llm' not in globals():
    _parse_request_llm = parse_request

def parse_request(text):
    info = _parse_request_llm(text) if llm_client is not None else None
    if info:
        return info
    print("(Gemini not available: the description is read with simple rules)")
    return parse_request_rules(text)

ANALYST_SYSTEM = """You are the Analyst of an Airbnb concierge service in Bordeaux.
Your job: gather evidence about one property and its market, using ONLY the tools provided.
Call get_property_profile, find_comparables, predict_price_range, summarise_reviews, and check_availability if dates are given.
Never invent a number: every number must come from a tool result. Cite the source of each claim.
Answer in JSON with keys: market_summary (2-3 sentences), comparables (from find_comparables), model (from predict_price_range),
availability (from check_availability or null), guest_insights (list of short strings, each with its listing_id source),
key_facts (list of {"claim": ..., "source": ...}).""" + STYLE


class Analyst:
    name = "Analyst"

    def fallback(self, file):
        comps = find_comparables(1.0)
        model = predict_price_range()
        avail = check_availability()
        rev = summarise_reviews()
        insights = [f"{e['theme']}: \"{e['excerpt']}\" (listing {e['listing_id']})" for e in rev['excerpts']]
        facts = [
            {'claim': f"{comps['n_comparables']} comparables within {comps['radius_km']} km, median €{comps['median']}",
             'source': comps['source']},
            {'claim': f"Model price €{model['predicted_price']}, 80% range €{model['range_80pct'][0]} to €{model['range_80pct'][1]}",
             'source': model['source']},
        ] + [{'claim': f"Guests nearby mention {t} more than in Bordeaux overall", 'source': rev['source']}
             for t in rev['themes_above_average']]
        summary = (f"{comps['n_comparables']} similar listings within {comps['radius_km']} km, priced €{comps['p25']} to €{comps['p75']} "
                   f"(P25 to P75). The model predicts €{model['predicted_price']}.")
        return {'market_summary': summary, 'comparables': comps, 'model': model,
                'availability': avail if avail.get('dates') else None, 'guest_insights': insights, 'key_facts': facts}

    def run(self, file):
        file.start(self.name)
        prompt = ("Analyse the current property." + (f" Requested dates: {file.data['dates']}." if file.data['dates'] else ' No dates.')
                  + (f" Manager's request: {file.data['request']}" if file.data['request'] else ''))
        answer = ask_llm(ANALYST_SYSTEM, prompt, tools=ANALYST_TOOLS, **llm_params('Analyst'))
        if answer and 'comparables' in answer and 'model' in answer:
            # The numbers are re-attached from the tools, so the LLM cannot alter them
            answer['comparables'] = find_comparables(answer['comparables'].get('radius_km', 1.0) if isinstance(answer['comparables'], dict) else 1.0)
            answer['model'] = predict_price_range()
            answer['availability'] = check_availability() if file.data['dates'] else None
            version = 'LLM'
        else:
            answer, version = self.fallback(file), 'fallback'
        file.data['evidence'] = answer
        c, m = answer['comparables'], answer['model']
        file.write(f"{self.name} ({version})", f"{c['n_comparables']} comparables within {c['radius_km']} km "
                   f"(P25 to P75: €{c['p25']} to €{c['p75']}); model €{m['predicted_price']}, range €{m['range_80pct'][0]} to €{m['range_80pct'][1]}")
        return file

STRATEGIST_SYSTEM = """You are the Strategist of an Airbnb concierge service in Bordeaux.
From the evidence brief and the property profile, decide:
1. segments: 2-4 guest segments relevant in this area, each with the evidence supporting it;
2. target: copy the exact label of one segment from segments; explain the choice in positioning and arguments;
3. positioning: one sentence (for [target], this property is the [frame of reference] that [benefit] because [reason to believe]);
4. value_proposition: 2-3 short benefits, each tied to a REAL feature of the property;
5. recommended_price (number, euros per night) and price_rationale: stay within the model's 80% range or the comparables' P10-P90,
   unless you explain why; if fewer than 30% of comparables are free on the dates, a higher price is defensible;
6. min_stay (nights) from the comparables' minimum stay if dates are given, else null;
7. arguments: list of {"argument": ..., "evidence": ...}, using only features the property has.
The highlights given by the manager are facts: use them in the positioning and the arguments.
Respect the manager brand brief and business objective; treat these as stated requirements, not market research.
If similar past cases are given (memory), stay consistent with them or explain the difference.
If the Validator gave feedback, fix it. Answer in JSON with exactly these keys.""" + STYLE


class Strategist:
    name = "Strategist"

    def fallback(self, file):
        f, e = file.data['features'], file.data['evidence']
        c, m, av = e['comparables'], e['model'], e.get('availability')
        if f['accommodates'] >= 4 and f['bedrooms'] >= 2:
            target = 'families'
        elif f['workspace'] and f['wifi'] and f['accommodates'] <= 2:
            target = 'business travellers'
        elif f['accommodates'] <= 2:
            target = 'couples'
        else:
            target = 'groups of friends'
        base_price = 0.5 * c['weighted_average_top20'] + 0.5 * m['predicted_price']
        coef = 1.0
        if av:
            if av['share_comparables_free'] < COMPETITION_THRESHOLDS[0]:
                coef = COMPETITION_ADJUSTMENT[0]
            elif av['share_comparables_free'] > COMPETITION_THRESHOLDS[1]:
                coef = COMPETITION_ADJUSTMENT[1]
        labels = {'air_conditioning': 'air conditioning', 'parking': 'parking', 'washer': 'washing machine',
                  'workspace': 'dedicated workspace', 'separate_kitchen': 'separate kitchen'}
        drivers = [d for d in m['drivers'] if d['effect_pct'] > 0 and d['value'] > 0 and d['feature'] in labels]
        arguments = [{'argument': labels[d['feature']], 'evidence': f"+{d['effect_pct']}% in the price model (SHAP)"} for d in drivers]
        if f['dist_centre_km'] <= 1.5:
            arguments.append({'argument': f"{f['dist_centre_km']} km from the city centre", 'evidence': 'property location'})
        for h in file.data.get('highlights', [])[:2]:
            arguments.append({'argument': h, 'evidence': 'stated by the manager'})
        if f['n_safety'] >= 2:
            arguments.append({'argument': 'safety equipment: ' + ', '.join(f['safety_items']), 'evidence': 'property amenities'})
        if not arguments:
            arguments.append({'argument': f"Capacity for {f['accommodates']} guests", 'evidence': 'property amenities'})
        return {
            'segments': [
                {'segment': 'families', 'evidence': f"Property-fit hypothesis: {f['accommodates']} guests and {int(f['bedrooms'])} bedrooms. " + ('Capacity supports this candidate.' if f['accommodates'] >= 4 and f['bedrooms'] >= 2 else 'Capacity limits this candidate.')},
                {'segment': 'business travellers', 'evidence': 'Property-fit hypothesis: ' + ('wifi and a workspace confirmed.' if f['wifi'] and f['workspace'] else 'wifi or a dedicated workspace is not confirmed.')},
                {'segment': 'couples', 'evidence': f"Property-fit hypothesis: space for {f['accommodates']} guests. " + ('Compact capacity fits a two-person stay.' if f['accommodates'] <= 2 else 'Larger capacity; willingness to pay needs validation.')},
                {'segment': 'groups of friends', 'evidence': f"Property-fit hypothesis: {f['accommodates']} guests, {f['beds']} beds. Actual traveller demand is not observed."},
            ],
            'target': target,
            'positioning': f"For {target}, a {('well-equipped' if c['property_better_equipped_than_share'] >= .5 else 'good-value')} "
                           f"base in Bordeaux, priced in line with {c['n_comparables']} similar listings nearby.",
            'value_proposition': [a['argument'] for a in arguments[:3]],
            'recommended_price': round(base_price * coef),
            'price_rationale': f"Half the top-20 comparables (€{c['weighted_average_top20']}), half the model (€{m['predicted_price']})"
                               + (f", ×{coef} for competition on the dates" if coef != 1 else ''),
            'min_stay': (av or {}).get('median_min_stay_nights'),
            'arguments': arguments,
        }

    def run(self, file):
        file.start(self.name)
        file.data.setdefault('highlights', [])
        feedback = file.data['feedback'].get(self.name)
        prompt = (f"Property profile, manager highlights, evidence and similar past cases: "
                  f"{file.compact('features', 'highlights', 'evidence', 'memory', 'brand_brief', 'objective')}"
                  + (f"\nValidator feedback to fix: {feedback}" if feedback else ''))
        answer = ask_llm(STRATEGIST_SYSTEM, prompt, **llm_params('Strategist'))
        needed = {'target', 'positioning', 'recommended_price', 'arguments'}
        if answer and needed <= set(answer):
            version = 'LLM'
        else:
            answer, version = self.fallback(file), 'fallback'
        file.data['strategy'] = answer
        file.write(f"{self.name} ({version})", f"Target: {answer['target']} | Price: €{answer['recommended_price']} | "
                                               f"{len(answer['arguments'])} argument(s)")
        return file

CREATIVE_SYSTEM = """You are the Creative lead of an Airbnb concierge service in Bordeaux.
Turn the approved strategy into execution, in English:
title (max 60 characters), description (max 100 words), campaign_concept (name + one sentence),
key_messages (3 short messages for the target), photo_directions (3-4 shots to take, in order),
social_post (max 40 words).
Use ONLY the strategy's arguments, the property's features and the manager's highlights, which you must showcase.
Never mention an amenity, a view or a service that does not appear in this information.
Do not mention the price. If the Validator gave feedback, fix it. Answer in JSON with exactly these keys.""" + STYLE_CREATIVE


class Creative:
    name = "Creative"
    HOOK = {'families': 'Family-friendly', 'business travellers': 'Business-ready', 'couples': 'Cosy', 'groups of friends': 'Spacious'}

    def fallback(self, file):
        f, s = file.data['features'], file.data['strategy']
        kind = 'flat' if f['room_type'] == 'Entire home/apt' else 'room'
        args = [a['argument'] for a in s['arguments']]
        args += [h for h in file.data.get('highlights', []) if h not in args]
        hook = self.HOOK.get(s['target'], 'Comfortable')
        return {
            'title': f"{hook} {kind} for {f['accommodates']} in Bordeaux",
            'description': f"Ideal for {s['target']}. This {kind} welcomes up to {f['accommodates']} guests."
                           + (f" Highlights: {'; '.join(args)}." if args else '')
                           + (f" Minimum stay: {s['min_stay']:.0f} nights." if s.get('min_stay') else ''),
            'campaign_concept': f"'{hook} Bordeaux': one message for {s['target']}: {args[0] if args else 'a well-located base'}.",
            'key_messages': ([f"Made for {s['target']}"] + args)[:3],
            'photo_directions': ['Main living space in daylight']
                                + [f"Close-up: {a}" for a in args if 'km from' not in a and not a.startswith('safety')][:2]
                                + [('Sleeping area' if f['bedrooms'] == 0 else 'Bedroom')] + (['Street or nearby landmark, to show the location'] if any('km from' in a for a in args) else []),
            'social_post': f"{hook} {kind} in Bordeaux for up to {f['accommodates']} guests. Book now!",
        }

    def run(self, file):
        file.start(self.name)
        feedback = file.data['feedback'].get(self.name)
        prompt = (f"Property features: {json.dumps({k: v for k, v in file.data['features'].items() if k != 'amenity_list'}, default=str)}\n"
                  f"Amenities: {file.data['features']['amenity_list']}\n"
                  f"Manager highlights (to showcase): {'; '.join(file.data.get('highlights') or []) or 'none'}\n"
                  f"Brand brief: {file.data.get('brand_brief', '')}\nObjective: {file.data.get('objective', '')}\nApproved strategy: {file.compact('strategy')}"
                  + (f"\nValidator feedback to fix: {feedback}" if feedback else ''))
        answer = ask_llm(CREATIVE_SYSTEM, prompt, **llm_params('Creative'))
        if answer and {'title', 'description', 'campaign_concept'} <= set(answer):
            version = 'LLM'
        else:
            answer, version = self.fallback(file), 'fallback'
        file.data['creative'] = answer
        file.write(f"{self.name} ({version})", f"Title: {answer['title']}")
        return file

AMENITY_WORDS = {'parking': 'parking', 'garage': 'parking', 'air conditioning': 'air_conditioning',
                 'air-conditioned': 'air_conditioning', 'a/c': 'air_conditioning', 'washing machine': 'washer',
                 'washer': 'washer', 'workspace': 'workspace', 'desk': 'workspace', 'separate kitchen': 'separate_kitchen'}

VALIDATOR_SYSTEM = """You are the Validator of an Airbnb concierge service. Check the consistency between
the evidence, the strategy and the creative output. Report only real problems, for example: a claim not supported by the evidence,
a target that does not match the property, a creative message that contradicts the positioning, an argument without evidence.
Answer in JSON: {"issues": [{"agent": "Strategist" or "Creative", "issue": "..."}]} (empty list if everything is consistent).
The highlights given by the manager are established facts: do not report them as unsupported.""" + STYLE


class Validator:
    name = "Validator"

    def rule_checks(self, file):
        f, e, s, c = file.data['features'], file.data['evidence'], file.data['strategy'], file.data['creative']
        comp, m = e['comparables'], e['model']
        price = float(s['recommended_price'])
        low, high = min(comp['p10'], m['range_80pct'][0]), max(comp['p90'], m['range_80pct'][1])
        text = ' '.join(str(c.get(k, '')) for k in ['title', 'description', 'social_post', 'campaign_concept']
                        ).lower() + ' ' + ' '.join(map(str, c.get('key_messages', []))).lower()
        said = ' '.join(file.data.get('highlights', [])).lower()   # what the manager stated
        absent = sorted({w for w, var in AMENITY_WORDS.items() if w in text and not f[var] and w not in said})
        issues = []
        if not low <= price <= high:
            issues.append({'agent': 'Strategist', 'issue': f"price €{price:.0f} outside the evidence range €{low:.0f} to €{high:.0f}"})
        if price > comp['median'] and not s.get('arguments'):
            issues.append({'agent': 'Strategist', 'issue': 'price above the comparables median with no argument'})
        if 'famil' in str(s['target']).lower() and (f['accommodates'] < 4 or f['bedrooms'] < 2):
            issues.append({'agent': 'Strategist', 'issue': 'family target but fewer than 4 guests or 2 bedrooms'})
        if 'business' in str(s['target']).lower() and not (f['workspace'] and f['wifi']):
            issues.append({'agent': 'Strategist', 'issue': 'business target without workspace and wifi'})
        if absent:
            issues.append({'agent': 'Creative', 'issue': f"mentions amenities the property does not have: {absent}"})
        if re.search(r'€\s?\d|\d\s?€|\d+\s?euros?', text):
            issues.append({'agent': 'Creative', 'issue': 'the creative text mentions a price'})
        return issues

    def run(self, file):
        file.start(self.name)
        issues = self.rule_checks(file)
        file.data.setdefault('highlights', [])
        review = ask_llm(VALIDATOR_SYSTEM, file.compact('features', 'highlights', 'evidence', 'strategy', 'creative'), **llm_params('Validator'))
        llm_issues = [i for i in (review or {}).get('issues', []) if i.get('agent') in ('Strategist', 'Creative')]
        issues += [dict(i, by='LLM review') for i in llm_issues]
        file.data['validation'] = {'status': 'VALIDATED' if not issues else 'TO REVIEW', 'issues': issues,
                                   'llm_review': review is not None}
        file.write(self.name, f"**{file.data['validation']['status']}**: "
                              + ('; '.join(i['agent'] + ': ' + i['issue'] for i in issues) or 'no issue'))
        return file


planner, analyst, strategist, creative, validator = Planner(), Analyst(), Strategist(), Creative(), Validator()

import html, inspect, re, tempfile
import plotly.graph_objects as go

# ---------------------------------------------------------------- style
COLORS = {'Planner': '#6C5CE7', 'Analyst': '#0984E3', 'Strategist': '#E17055', 'Creative': '#00B894',
          'Validator': '#2D3436', 'Manager': '#FD79A8'}
ICONS = {'Planner': 'P', 'Analyst': 'A', 'Strategist': 'S', 'Creative': 'C', 'Validator': 'V', 'Manager': 'M'}
ROLES = {a: AGENT_CONFIG[a]['role'] for a in AGENT_CONFIG}
AGENTS = ['Planner', 'Analyst', 'Strategist', 'Creative', 'Validator']
CORAL = '#FF385C'

CSS = """
.gradio-container {max-width: 1400px !important; margin: auto;}
.date-row, .date-row .block, .date-row .form {overflow: visible !important; z-index: 50;}
body, gradio-app, .gradio-container {background: linear-gradient(160deg, #fff1f4 0%, #f5f0ff 50%, #eef6ff 100%) !important;}
.block, .form {border-radius: 14px !important; box-shadow: 0 2px 12px rgba(108,92,231,.06);}
.hero {background: linear-gradient(120deg, #FF385C 0%, #E61E4D 45%, #6C5CE7 100%); border-radius: 18px;
       padding: 26px 30px; color: white; box-shadow: 0 10px 30px rgba(230,30,77,.25);}
.hero h1 {margin: 0; font-size: 30px; color: white;} .hero p {margin: 6px 0 0; opacity: .92; font-size: 15px;}
.pill {display:inline-block; background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
       padding: 4px 12px; border-radius: 999px; margin: 12px 8px 0 0; font-size: 13px;}
.agent-card {display:flex; gap:12px; align-items:flex-start; padding:12px 14px; border-radius:14px; margin-bottom:10px;
             background:#fff; border:1px solid #eceef1; transition: all .3s;}
.agent-card.running {border-color:#0984E3; box-shadow:0 0 0 3px rgba(9,132,227,.15); animation: pulse 1.2s infinite;}
.agent-card.done {border-left: 5px solid #00B894;}
.agent-card.warn {border-left: 5px solid #E17055;}
@keyframes pulse {0% {box-shadow:0 0 0 0 rgba(9,132,227,.35);} 70% {box-shadow:0 0 0 10px rgba(9,132,227,0);} 100% {box-shadow:0 0 0 0 rgba(9,132,227,0);}}
.avatar {width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:17px; font-weight:800; flex-shrink:0;}
.stepper {display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:10px; margin:6px 0 14px}
.stepper .agent-card {flex-direction:column; gap:6px; margin:0; padding:10px 12px}
.step-n {font-size:11px; color:#999; font-weight:700}
.warn-box {background:#fff4e5; border:1px solid #f5c26b; color:#8a5300; border-radius:12px; padding:10px 14px; margin-bottom:12px; font-size:14px}
.spec {font-size:12px; color:#555; margin-top:6px} .spec summary {cursor:pointer; color:#0984E3}
.kpi {background:#fff; border:1px solid #eceef1; border-radius:16px; padding:16px; box-shadow:0 2px 10px rgba(0,0,0,.04);}
.kpi .v {font-size:28px; font-weight:800; color:#222;} .kpi .l {font-size:12px; color:#717171; text-transform:uppercase; letter-spacing:.5px;}
.bubble {padding:10px 14px; border-radius:14px; background:#f7f7f9; margin:4px 0 12px; font-size:14px; line-height:1.45;}
"""

def to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        m = re.search(r'\d+(?:\.\d+)?', str(x))
        return float(m.group()) if m else default

def esc(x):
    return html.escape(str(x)) if x is not None else ''

def md(x):
    # Escape, then render the small Markdown used by the LLMs (**bold**, *italic*, `code`)
    t = esc(x)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<i>\1</i>', t)
    return re.sub(r'`(.+?)`', r'<code>\1</code>', t)

def llm_badge():
    if globals().get('llm_client') is None:
        return 'Rule-based mode, Gemini off'
    return f"Gemini · {globals().get('ACTIVE_MODEL', GEMINI_MODEL)}"

def hero_html():
    return f"""<div class='hero'><h1>Airbnb Concierge Adviser, Bordeaux</h1>
    <p>Five AI agents turn a property description into a market-backed price, a target and a ready-to-publish listing, under your control.</p>
    <span class='pill'>{esc(llm_badge())}</span><span class='pill'>Inside Airbnb data, Bordeaux, September 2026</span></div>"""

# ---------------------------------------------------------------- agent pipeline & conversation
def pipeline_html(status):
    # status: {agent: (state, text)} with state in wait / run / done / warn
    labels = {'wait': ('Waiting', '#b2bec3'), 'run': ('Working', '#0984E3'),
              'done': ('Done', '#00B894'), 'warn': ('Your turn', '#E17055')}
    out = ["<div class='stepper' translate='no'>"]
    for n, a in enumerate(AGENTS, 1):
        state, text = status.get(a, ('wait', ''))
        lab, col = labels[state]
        short = text if len(text) <= 140 else text[:137] + '…'
        out.append(f"""<div class='agent-card {state}'>
          <div style='display:flex; align-items:center; gap:8px'>
            <div class='avatar' style='background:{COLORS[a]}22; color:{COLORS[a]}; width:32px; height:32px; font-size:14px'>{ICONS[a]}</div>
            <div><div class='step-n'>STEP {n}</div><b style='color:{COLORS[a]}'>{a}</b></div></div>
          <span style='align-self:flex-start; font-size:11px; font-weight:700; color:white; background:{col}; padding:2px 9px; border-radius:999px'>{lab}</span>
          <div style='font-size:12px; color:#333'>{md(short) if short else esc(ROLES[a])}</div>
          <details class='spec'><summary>Input, tools, output</summary>
            <b>LLM</b>: Gemini, T={AGENT_CONFIG[a]['llm']['temperature']}, max {AGENT_CONFIG[a]['llm']['max_tokens']} tokens<br>
            <b>Input</b>: {esc(AGENT_CONFIG[a]['input'])}<br><b>Tools</b>: {esc(', '.join(AGENT_CONFIG[a]['tools']))}<br>
            <b>Output</b>: {esc(AGENT_CONFIG[a]['output'])}</details></div>""")
    return ''.join(out) + '</div>'

def empty_fig(message):
    # Empty chart with a message, until results arrive
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14, color='#999'))
    fig.update_layout(height=260, plot_bgcolor='#fafafa', paper_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

PLACEHOLDERS = {'map': 'The map of comparables will appear here', 'price': 'The price position in the market will appear here',
                'drivers': 'The price drivers (SHAP) will appear here', 'reviews': 'The guest voice will appear here'}

def ph(kind):
    return empty_fig(PLACEHOLDERS[kind])

def norm_date(value):
    # Accepts 2026-12-19 or 19/12/2026; returns YYYY-MM-DD, or None if the date is empty or unreadable
    value = str(value or '').strip()
    if not value:
        return None
    try:
        return pd.to_datetime(value, dayfirst='/' in value).strftime('%Y-%m-%d')
    except Exception:
        return None

def size_warnings(guests, bedrooms, beds, bathrooms):
    # Beyond the 99th percentile of the data, the model has seen very few similar properties
    limits = base[['accommodates', 'bedrooms', 'beds', 'n_bathrooms']].quantile(0.99)
    checks = [('guests', guests, limits['accommodates']), ('bedrooms', bedrooms, limits['bedrooms']),
              ('beds', beds, limits['beds']), ('bathrooms', bathrooms, limits['n_bathrooms'])]
    return [f"{value:g} {name}: very few properties of this size in Bordeaux (99% have at most {limit:g}), so the price is less reliable."
            for name, value, limit in checks if value > limit]

def feed_html(log):
    if not log:
        return "<div style='color:#888'>The agents' conversation will appear here.</div>"
    out = []
    for line in log:
        m = re.match(r'^\[(.+?)\]\s*(.*)$', line)
        who, msg = (m.group(1), m.group(2)) if m else ('Planner', line)
        name = who.split(' (')[0]
        mode = who[len(name):].strip(' ()')
        tag = (f"<span style='font-size:10px; background:{'#6C5CE7' if mode == 'LLM' else '#b2bec3'}; color:white; "
               f"padding:1px 7px; border-radius:999px; margin-left:6px'>{'Gemini' if mode == 'LLM' else 'rules'}</span>") if mode else ''
        col = COLORS.get(name, '#636e72')
        if msg.startswith('starts'):
            out.append(f"<div style='font-size:12px; color:#888; margin:2px 0 6px 44px'><b style='color:{col}'>{esc(name)}</b> {esc(msg)}</div>")
            continue
        out.append(f"""<div style='display:flex; gap:10px'>
          <div class='avatar' style='background:{col}22; color:{col}; width:34px; height:34px; font-size:14px'>{ICONS.get(name, '·')}</div>
          <div style='flex:1'><b style='color:{col}'>{esc(name)}</b>{tag}<div class='bubble'>{md(msg)}</div></div></div>""")
    return "<div style='max-height:560px; overflow-y:auto; padding-right:6px'>" + ''.join(out) + "</div>"

def last_message(file, agent):
    for line in reversed(file.data['log']):
        if line.startswith('[' + agent):
            return line.split('] ', 1)[-1][:160]
    return ''

# ---------------------------------------------------------------- evidence views
def kpi_html(file):
    e, s = file.data['evidence'], file.data.get('strategy') or {}
    c, m, av = e.get('comparables', {}), e.get('model', {}), e.get('availability')
    rng = m.get('range_80pct', ['?', '?'])
    cards = [
        ('Recommended price', f"€{to_float(s['recommended_price']):.0f}" if s.get('recommended_price') is not None else '…', 'per night', CORAL),
        ('Model estimate', f"€{m.get('predicted_price', '?')}", f"80% range: €{rng[0]} to €{rng[1]}", '#0984E3'),
        ('Comparables', f"{c.get('n_comparables', '?')}", f"within {c.get('radius_km', '?')} km · median €{c.get('median', '?')}", '#6C5CE7'),
        ('Competition', f"{av['share_comparables_free']:.0%} free" if av else 'No dates',
         f"{av['n_comparables_free']} competitors still free" if av else 'add dates to see it', '#00B894'),
    ]
    warn = ''.join(f"<div class='warn-box'>{esc(w)}</div>" for w in file.data.get('warnings', []))
    return warn + "<div style='display:grid; grid-template-columns:repeat(4,1fr); gap:12px'>" + ''.join(
        f"<div class='kpi' style='border-top:4px solid {col}'><div class='l'>{l}</div><div class='v'>{v}</div>"
        f"<div style='font-size:12px; color:#717171'>{sub}</div></div>" for l, v, sub, col in cards) + "</div>"

def map_figure(file):
    comps, f = CURRENT.get('comparables'), file.data['features']
    if comps is None or len(comps) == 0:
        return ph('map')
    use_new = hasattr(go, 'Scattermap')
    Trace = go.Scattermap if use_new else go.Scattermapbox
    top = comps.nlargest(N_TOP_COMPARABLES, 'score')
    fig = go.Figure()
    fig.add_trace(Trace(lat=comps['latitude'], lon=comps['longitude'], mode='markers', name='Comparables',
                        marker=dict(size=8, color=comps['price'].clip(upper=comps['price'].quantile(.95)),
                                    colorscale='Sunsetdark', showscale=True, colorbar=dict(title='€/night'), opacity=.75),
                        text=[f"€{p:.0f} · {a} guests · {d:.2f} km" for p, a, d in zip(comps['price'], comps['accommodates'], comps['distance_km'])],
                        hoverinfo='text'))
    fig.add_trace(Trace(lat=top['latitude'], lon=top['longitude'], mode='markers', name=f'Top {len(top)} most similar',
                        marker=dict(size=13, color='#6C5CE7', opacity=.9),
                        text=[f"Top match · €{p:.0f} · score {s:.2f}" for p, s in zip(top['price'], top['score'])], hoverinfo='text'))
    fig.add_trace(Trace(lat=[f['latitude']], lon=[f['longitude']], mode='markers', name='Your property',
                        marker=dict(size=22, color=CORAL), text=['Your property'], hoverinfo='text'))
    radius = file.data['evidence']['comparables'].get('radius_km', 1)
    view = dict(style='open-street-map', center=dict(lat=f['latitude'], lon=f['longitude']),
                zoom=14 if radius <= 1 else 13 if radius <= 2 else 12)
    fig.update_layout(**({'map': view} if use_new else {'mapbox': view}), height=430, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation='h', y=1.02, x=0, bgcolor='rgba(255,255,255,.8)'))
    return fig

def price_figure(file):
    comps = CURRENT.get('comparables')
    if comps is None or len(comps) == 0:
        return ph('price')
    m, s = file.data['evidence']['model'], file.data.get('strategy') or {}
    prices = comps['price'][comps['price'] <= comps['price'].quantile(.97)]
    fig = go.Figure(go.Histogram(x=prices, nbinsx=30, marker_color='#dfe6e9', name='Comparables'))
    lo, hi = m['range_80pct']
    fig.add_vrect(x0=lo, x1=hi, fillcolor='#0984E3', opacity=.12, line_width=0,
                  annotation_text='Model 80% range', annotation_position='top left')
    fig.add_vline(x=m['predicted_price'], line_dash='dash', line_color='#0984E3',
                  annotation_text=f"Model €{m['predicted_price']:.0f}", annotation_position='top')
    if s.get('recommended_price') is not None:
        rec = to_float(s['recommended_price'])
        fig.add_vline(x=rec, line_width=4, line_color=CORAL,
                      annotation_text=f"Recommended €{rec:.0f}", annotation_position='top right')
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', showlegend=False,
                      xaxis_title='Price per night (€)', yaxis_title='Number of comparables',
                      title=dict(text='Where the price sits in the local market', font=dict(size=15)))
    return fig

def drivers_figure(file):
    d = file.data['evidence'].get('model', {}).get('drivers') or []
    if not d:
        return ph('drivers')
    d = sorted(d, key=lambda x: x['effect_pct'])
    fig = go.Figure(go.Bar(x=[x['effect_pct'] for x in d], y=[FEATURE_LABELS.get(x['feature'], x['feature'].replace('_', ' ')) for x in d], orientation='h',
                           marker_color=['#00B894' if x['effect_pct'] >= 0 else '#E17055' for x in d],
                           text=[f"{x['effect_pct']:+.1f}%" for x in d], textposition='outside'))
    fig.update_layout(height=320, margin=dict(l=10, r=30, t=40, b=10), plot_bgcolor='white',
                      title=dict(text='Why this price? (SHAP drivers)', font=dict(size=15)), xaxis_title='Effect on price (%)')
    return fig

def reviews_figure():
    try:
        themes = summarise_reviews().get('themes')
    except Exception:
        return ph('reviews')
    if not themes:
        return ph('reviews')
    names = list(themes)
    labels = names
    fig = go.Figure([go.Bar(name='Nearby comparables', x=labels, y=[themes[n]['nearby_pct'] for n in names], marker_color=CORAL),
                     go.Bar(name='Bordeaux average', x=labels, y=[themes[n]['bordeaux_pct'] for n in names], marker_color='#b2bec3')])
    fig.update_layout(barmode='group', height=320, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white',
                      title=dict(text='Guest voice: what nearby reviews talk about (NLP)', font=dict(size=15)),
                      yaxis_title='% of reviews', legend=dict(orientation='h', y=1.12, x=0))
    return fig

def insights_html(file):
    e = file.data['evidence']
    facts = ''.join(f"<li>{md(k.get('claim'))} <span style='color:#999; font-size:12px'>[{esc(k.get('source'))}]</span></li>"
                    for k in e.get('key_facts', []) if isinstance(k, dict))
    quotes = ''.join(f"<div class='bubble' style='background:#fff4f6'>{md(q)}</div>" for q in e.get('guest_insights', [])[:4])
    return (f"<div class='kpi'><b>Market summary</b><p>{md(e.get('market_summary'))}</p>"
            f"<b>Key facts (every number has a source)</b><ul>{facts}</ul><b>What guests say nearby</b>{quotes}</div>")

def strategy_html(file):
    s = file.data.get('strategy') or {}
    if not s:
        return "<div style='color:#888'>The Strategist's proposal will appear here.</div>"
    args = s.get('arguments', [])
    chips = ''.join(
        f"<div style='padding:8px 12px; border-radius:12px; background:#f7f7f9; margin:6px 0'><b>{md(a.get('argument') if isinstance(a, dict) else a)}</b>"
        f"<span style='color:#717171; font-size:12px'> : {md(a.get('evidence', '')) if isinstance(a, dict) else ''}</span></div>" for a in args)
    return f"""<div class='kpi' style='border-top:4px solid {COLORS['Strategist']}'>
      <div class='l'>Proposed target</div><div class='v' style='color:{COLORS['Strategist']}'>{esc(s.get('target'))}</div>
      <p style='font-size:15px'>{md(s.get('positioning'))}</p>
      <div style='display:flex; gap:24px; margin:10px 0'>
        <div><div class='l'>Price per night</div><div class='v'>€{to_float(s.get('recommended_price')):.0f}</div></div>
        <div><div class='l'>Minimum stay</div><div class='v'>{esc(s.get('min_stay') or 'not set')}</div></div></div>
      <div style='font-size:13px; color:#555'><b>Price rationale:</b> {md(s.get('price_rationale', ''))}</div>
      <div style='margin-top:12px'><b>Arguments backed by the data</b>{chips}</div></div>
      <div style='margin-top:10px; padding:12px; border-radius:12px; background:#fff8e1; font-size:14px'>
      <b>Your turn:</b> edit the target or the price below if needed, then approve so the Creative agent writes the listing.</div>"""

def listing_html(file, approved=False):
    c, v, s = file.data.get('creative') or {}, file.data.get('validation') or {}, file.data.get('strategy') or {}
    if not c:
        return "<div style='color:#888'>Approve the strategy to generate the listing.</div>"
    ok = v.get('status') == 'VALIDATED'
    badge = (f"<span style='background:{'#00B894' if ok else '#E17055'}; color:white; padding:4px 12px; border-radius:999px; "
             f"font-weight:700; font-size:12px'>{'VALIDATED' if ok else 'TO REVIEW'}</span>")
    stamp = ("<div style='margin-top:12px; padding:12px; border-radius:12px; background:#e8f8f2; color:#00795a; font-weight:700'>"
             "Approved by the manager, ready to publish</div>") if approved else ''
    issues = ''.join(f"<li><b>{esc(i.get('agent'))}</b>: {md(i.get('issue'))}</li>" for i in v.get('issues', []))
    lst = lambda xs: ''.join(f"<li>{md(x)}</li>" for x in (xs or []))
    return f"""<div style='display:grid; grid-template-columns: 3fr 2fr; gap:16px'>
      <div class='kpi' style='padding:0; overflow:hidden'>
        <div style='height:150px; background:linear-gradient(135deg,#ffd1dc,#c8b6ff,#a0e7e5); display:flex; align-items:flex-end; padding:16px'>
          <span style='background:white; padding:4px 10px; border-radius:8px; font-weight:700'>Bordeaux · {esc(s.get('target'))}</span></div>
        <div style='padding:18px'><div style='display:flex; justify-content:space-between; gap:10px'>
          <h2 style='margin:0; font-size:22px'>{esc(c.get('title'))}</h2>{badge}</div>
          <div style='margin:8px 0; font-size:20px'><b>€{to_float(s.get('recommended_price')):.0f}</b> <span style='color:#717171; font-size:14px'>night</span></div>
          <p style='font-size:15px; line-height:1.55'>{md(c.get('description'))}</p>
          {f"<div style='color:#E17055'><b>Issues found by the Validator</b><ul>{issues}</ul></div>" if issues else ''}{stamp}</div></div>
      <div>
        <div class='kpi'><b>Campaign concept</b><p>{md(c.get('campaign_concept'))}</p>
          <b>Key messages</b><ul>{lst(c.get('key_messages'))}</ul>
          <b>Photo directions</b><ol>{lst(c.get('photo_directions'))}</ol></div>
        <div class='kpi' style='margin-top:12px; background:#111; color:white'><b>Social post</b>
          <p style='font-size:14px'>{md(c.get('social_post'))}</p></div></div></div>"""

# ---------------------------------------------------------------- the three steps (generators = live updates)
AMENITY_GROUPS = {
    'Comfort': ['Wifi', 'TV', 'Air conditioning', 'Heating', 'Elevator', 'Balcony'],
    'Kitchen and laundry': ['Kitchen', 'Dishwasher', 'Microwave', 'Refrigerator', 'Oven', 'Coffee maker', 'Washer'],
    'Family, work and car': ['Crib', 'Dedicated workspace', 'Free parking on premises'],
    'Safety': ['Smoke alarm', 'Carbon monoxide alarm', 'Fire extinguisher', 'First aid kit'],
    'Essentials': ['Hot water', 'Bed linens', 'Hair dryer', 'Iron'],
    'Everyday small items': ['Essentials', 'Hangers', 'Shampoo', 'Body soap', 'Dishes and silverware', 'Cooking basics',
                                        'Hot water kettle', 'Toaster', 'Freezer', 'Stove', 'Wine glasses', 'Dining table',
                                        'Extra pillows and blankets', 'Room-darkening shades', 'Clothing storage',
                                        'Drying rack for clothing', 'Self check-in', 'Luggage dropoff allowed'],
}
SMALL_ITEMS = AMENITY_GROUPS['Everyday small items']
KITCHEN_CHOICES = ['Separate kitchen', 'Open kitchen', 'Kitchenette / not sure', 'No kitchen']
FEATURE_LABELS = {'accommodates': 'guests', 'n_bathrooms': 'bathrooms', 'n_amenities': 'number of amenities',
                  'n_safety': 'safety items', 'air_conditioning': 'air conditioning', 'washer': 'washing machine',
                  'separate_kitchen': 'separate kitchen', 'dist_centre_km': 'distance to centre', 'no_rating': 'no rating yet'}
ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
AMENITY_LABELS = {'Essentials': 'Essentials (towels, sheets, toilet paper)'}
DEFAULT_NB = 'Centre ville (Bordeaux)' if 'Centre ville (Bordeaux)' in NEIGHBOURHOODS else NEIGHBOURHOODS[0]

def split_highlights(notes):
    # 'Quiet and close to shops, step-free access' → ['Quiet and close to shops', 'step-free access']
    return [h.strip(' .') for h in re.split(r'[,;\n]', notes or '') if h.strip(' .')]

def form_to_info(nb, room, guests, bedrooms, beds, bathrooms, surface, kitchen, notes, checkin, checkout, *groups):
    guests, bedrooms, beds, bathrooms = int(guests or 1), int(bedrooms or 0), int(beds or 1), float(bathrooms or 1)
    checkin, checkout = norm_date(checkin), norm_date(checkout)
    checked = [a for g in groups for a in (g or [])]
    if kitchen in ('Separate kitchen', 'Open kitchen', 'Kitchenette / not sure') and 'Kitchen' not in checked:
        checked.append('Kitchen')
    kitchen_text = {'Separate kitchen': 'Separate kitchen.', 'Open kitchen': 'Open kitchen.',
                    'Kitchenette / not sure': 'Kitchenette.', 'No kitchen': ''}[kitchen]
    notes = (notes or '').strip()
    if notes and notes[-1] not in '.!?':
        notes += '.'
    description = ' '.join(x for x in [f"{room} for {int(guests)} guests in {nb}.", kitchen_text, notes] if x)
    info = {'room_type': room, 'accommodates': int(guests), 'bedrooms': int(bedrooms), 'beds': int(beds),
            'bathrooms': float(bathrooms), 'surface_m2': surface or None, 'neighbourhood': nb, 'amenities': checked,
            'checkin': (checkin or '').strip() or None, 'checkout': (checkout or '').strip() or None, 'description': description}
    padded = checked
    request = (f"{description} Amenities: {', '.join(checked) or 'none listed'}."
               + (f" Dates: {info['checkin']} to {info['checkout']}." if info['checkin'] and info['checkout'] else ''))
    return info, padded, request

def run_analysis(nb, room, guests, bedrooms, beds, bathrooms, surface, kitchen, notes, checkin, checkout, *groups):
    status = {a: ('wait', '') for a in AGENTS}
    status['Planner'] = ('run', 'Reading the form…')
    yield (None, pipeline_html(status), feed_html([]), '', ph('map'), ph('price'), ph('drivers'), '', ph('reviews'), strategy_html(_Empty()), '', None,
           listing_html(_Empty()), hero_html())
    info, padded, request_text = form_to_info(nb, room, guests, bedrooms, beds, bathrooms, surface, kitchen,
                                              notes, checkin, checkout, *groups)
    dates = (info['checkin'], info['checkout']) if info['checkin'] and info['checkout'] else None
    prop = build_property(info)
    prop['amenities'] = json.dumps(padded)
    file = SharedPropertyFile(prop, dates, request=request_text)
    file.verbose = False
    file.write('Manager', request_text)
    file.data['highlights'] = split_highlights(notes)
    file.data['warnings'] = size_warnings(info['accommodates'], info['bedrooms'], info['beds'], info['bathrooms'])
    for w in file.data['warnings']:
        file.write('Planner', 'Warning: ' + w)
    file = planner.prepare(file)
    status['Planner'] = ('done', last_message(file, 'Planner'))
    status['Analyst'] = ('run', 'Finding comparables, running the price model…')
    yield (file, pipeline_html(status), feed_html(file.data['log']), '', ph('map'), ph('price'), ph('drivers'), '', ph('reviews'), strategy_html(file), '', None,
           listing_html(file), hero_html())
    file = analyst.run(file)
    voice = reviews_figure()
    status['Analyst'] = ('done', last_message(file, 'Analyst'))
    status['Strategist'] = ('run', 'Choosing the target and the price…')
    yield (file, pipeline_html(status), feed_html(file.data['log']), kpi_html(file), map_figure(file), price_figure(file),
           drivers_figure(file), insights_html(file), voice, strategy_html(file), '', None, listing_html(file), hero_html())
    file = strategist.run(file)
    s = file.data['strategy']
    status['Strategist'] = ('done', last_message(file, 'Strategist'))
    status['Creative'] = ('warn', 'Waiting for your approval of the strategy')
    yield (file, pipeline_html(status), feed_html(file.data['log']), kpi_html(file), map_figure(file), price_figure(file),
           drivers_figure(file), insights_html(file), voice, strategy_html(file), s['target'], to_float(s['recommended_price']),
           listing_html(file), hero_html())

def run_creation(file, target, price):
    if file is None:
        yield None, pipeline_html({}), feed_html([]), "<b>Run the analysis first.</b>", strategy_html(_Empty())
        return
    status = {a: ('done', last_message(file, a)) for a in ['Planner', 'Analyst', 'Strategist']}
    s = file.data['strategy']
    if target != s['target'] or to_float(price) != to_float(s['recommended_price']):
        file.write('Manager', f"edited the strategy: target {s['target']} → {target}, price €{s['recommended_price']} → €{price}")
        s['target'], s['recommended_price'] = target, to_float(price)
    else:
        file.write('Manager', 'approved the strategy')
    file.data['approvals'].append({'step': 'strategy', 'decision': 'approved', 'by': 'manager'})
    status['Creative'] = ('run', 'Writing the listing and the campaign…')
    yield file, pipeline_html(status), feed_html(file.data['log']), listing_html(file), strategy_html(file)
    file = creative.run(file)
    for round_ in range(MAX_REVISIONS + 1):
        status['Creative'] = ('done', last_message(file, 'Creative'))
        status['Validator'] = ('run', 'Checking every claim…')
        yield file, pipeline_html(status), feed_html(file.data['log']), listing_html(file), strategy_html(file)
        file = validator.run(file)
        issues = file.data['validation']['issues']
        if not issues or round_ == MAX_REVISIONS:
            break
        redo = {i['agent'] for i in issues}
        file.data['feedback'] = {a: [i['issue'] for i in issues if i['agent'] == a] for a in redo}
        file.write('Planner', f"Validator sent the work back to {sorted(redo)} (revision {round_ + 1})")
        status['Validator'] = ('warn', f"{len(issues)} issue(s), revision {round_ + 1}")
        status['Creative'] = ('run', 'Revising after the Validator feedback…')
        yield file, pipeline_html(status), feed_html(file.data['log']), listing_html(file), strategy_html(file)
        if 'Strategist' in redo:
            file = strategist.run(file)
        file = creative.run(file)
    v = file.data['validation']
    status['Validator'] = ('done' if v['status'] == 'VALIDATED' else 'warn', last_message(file, 'Validator'))
    yield file, pipeline_html(status), feed_html(file.data['log']), listing_html(file), strategy_html(file)

def approve_final(file):
    if file is None:
        return None, feed_html([]), "<b>Nothing to approve yet.</b>", None
    file.data['approvals'].append({'step': 'final listing', 'decision': 'approved', 'by': 'manager'})
    file.write('Manager', 'approved the final listing')
    MEMORY.save(file)
    path = tempfile.NamedTemporaryFile(delete=False, suffix='_property_file.json').name
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({k: v for k, v in file.data.items() if k != 'property'}, fh, ensure_ascii=False, indent=2, default=str)
    return file, feed_html(file.data['log']), listing_html(file, approved=True), path

class _Empty:
    data = {'strategy': {}, 'creative': {}, 'validation': {}}

# ---------------------------------------------------------------- What-if simulator (price model only: instant, no Gemini)
SIM_BINARIES = [('Air conditioning', 'air_conditioning'), ('Free parking', 'parking'), ('Washing machine', 'washer'),
                ('Dedicated workspace', 'workspace'), ('Separate kitchen', 'separate_kitchen')]

def sim_features(nb, room, guests, bedrooms, bathrooms, checks, n_safety, n_amenities):
    centre = NB_CENTRES.loc[nb] if nb in NB_CENTRES.index else pd.Series({'latitude': BORDEAUX_CENTRE[0], 'longitude': BORDEAUX_CENTRE[1]})
    f = {'room_type': room, 'neighbourhood': nb, 'accommodates': int(guests), 'bedrooms': float(bedrooms),
         'beds': float(max(1, round(guests / 2))), 'n_bathrooms': float(bathrooms), 'n_amenities': int(n_amenities),
         'n_safety': int(n_safety),
         'dist_centre_km': float(haversine_km(centre['latitude'], centre['longitude'], *BORDEAUX_CENTRE))}
    for label, var in SIM_BINARIES:
        f[var] = int(label in (checks or []))
    return f

def sim_predict(f, rating, new_listing, superhost):
    x = x_row_of({}, f)
    for col, val in [('superhost', float(superhost)), ('rating', float(rating)), ('no_rating', float(new_listing))]:
        if col in x.columns:
            x[col] = val
    return (float(np.exp(best_model.predict(x))[0]), float(np.exp(q_low.predict(x))[0]), float(np.exp(q_high.predict(x))[0]))

def sim_market(nb, room, guests):
    pool = base[(base['room_type'] == room) & (base['accommodates'].sub(guests).abs() <= 1)]
    local = pool[pool['neighbourhood_cleansed'] == nb]
    return (local, f'in {nb}') if len(local) >= 15 else (pool, 'in Bordeaux')

def simulate(nb, room, guests, bedrooms, bathrooms, checks, n_safety, n_amenities, rating, new_listing, superhost, reference):
    guests, bedrooms, bathrooms = int(guests or 1), int(bedrooms or 0), float(bathrooms or 1)
    n_amenities = int(n_amenities or 0)
    f = sim_features(nb, room, guests, bedrooms, bathrooms, checks, n_safety, n_amenities)
    price, lo, hi = sim_predict(f, rating, new_listing, superhost)
    ref = reference if reference else price
    delta = price - ref
    # Value of each lever: price with the change minus price without it, everything else unchanged
    levers = []
    for label, var in SIM_BINARIES:
        g_on, g_off = dict(f, **{var: 1}), dict(f, **{var: 0})
        gain = sim_predict(g_on, rating, new_listing, superhost)[0] - sim_predict(g_off, rating, new_listing, superhost)[0]
        levers.append((label, gain, bool(f[var])))
    for label, change in [('+1 safety item', {'n_safety': min(4, f['n_safety'] + 1)}), ('+5 amenities', {'n_amenities': f['n_amenities'] + 5}),
                          ('+1 guest', {'accommodates': f['accommodates'] + 1}), ('+1 bedroom', {'bedrooms': f['bedrooms'] + 1})]:
        levers.append((label, sim_predict(dict(f, **change), rating, new_listing, superhost)[0] - price, False))
    pool, where = sim_market(nb, room, guests)
    warn = ''.join(f"<div class='warn-box'>{esc(w)}</div>" for w in size_warnings(guests, bedrooms, max(1, round(guests / 2)), bathrooms))
    pct = float((pool['price'] < price).mean()) if len(pool) else np.nan

    arrow = '▲' if delta > 0.5 else '▼' if delta < -0.5 else '='
    col = '#00B894' if delta > 0.5 else '#E17055' if delta < -0.5 else '#717171'
    card = warn + f"""<div style='display:grid; grid-template-columns:2fr 1fr 1fr; gap:12px'>
      <div class='kpi' style='border-top:4px solid {CORAL}'><div class='l'>Predicted price per night</div>
        <div style='font-size:44px; font-weight:800'>€{price:.0f} <span style='font-size:20px; color:{col}'>{arrow} {delta:+.0f} €</span></div>
        <div style='color:#717171'>vs reference €{ref:.0f} · 80% range: €{lo:.0f} to €{hi:.0f}</div></div>
      <div class='kpi'><div class='l'>Market position</div><div class='v'>{pct:.0%}</div>
        <div style='font-size:12px; color:#717171'>of similar listings {esc(where)} are cheaper</div></div>
      <div class='kpi'><div class='l'>Similar listings</div><div class='v'>{len(pool)}</div>
        <div style='font-size:12px; color:#717171'>{esc(room)}, {int(guests)}±1 guests {esc(where)}</div></div></div>"""

    lev = sorted(levers, key=lambda t: t[1])
    fig_l = go.Figure(go.Bar(x=[g for _, g, _ in lev], y=[(l + (' ✓' if has else '')) for l, _, has in lev], orientation='h',
                             marker_color=['#b2bec3' if has else ('#00B894' if g >= 0 else '#E17055') for _, g, has in lev],
                             text=[f"{g:+.0f} €" for _, g, _ in lev], textposition='outside'))
    fig_l.update_layout(height=360, margin=dict(l=10, r=40, t=50, b=10), plot_bgcolor='white',
                        title=dict(text='What is each improvement worth per night? (✓ = already there)', font=dict(size=15)),
                        xaxis_title='Change in predicted price (€)')
    prices = pool['price'][pool['price'] <= pool['price'].quantile(.97)] if len(pool) else pd.Series(dtype=float)
    fig_m = go.Figure(go.Histogram(x=prices, nbinsx=30, marker_color='#dfe6e9'))
    fig_m.add_vrect(x0=lo, x1=hi, fillcolor='#0984E3', opacity=.12, line_width=0, annotation_text='80% range', annotation_position='top left')
    fig_m.add_vline(x=ref, line_dash='dot', line_color='#636e72', annotation_text=f'Reference €{ref:.0f}', annotation_position='bottom right')
    fig_m.add_vline(x=price, line_width=4, line_color=CORAL, annotation_text=f'Simulated €{price:.0f}', annotation_position='top right')
    fig_m.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white', showlegend=False,
                        title=dict(text=f'Position among similar listings {where}', font=dict(size=15)),
                        xaxis_title='Price per night (€)', yaxis_title='Listings')
    return card, fig_l, fig_m, ref

def load_into_simulator(file):
    if file is None:
        return [DEFAULT_NB, 'Entire home/apt', 4, 2, 1.0, [], 1, 25]
    f = file.data['features']
    checks = [label for label, var in SIM_BINARIES if f.get(var)]
    nb = f.get('neighbourhood') if f.get('neighbourhood') in NEIGHBOURHOODS else DEFAULT_NB
    return [nb, f['room_type'], int(f['accommodates']), int(f.get('bedrooms') or 1), float(f.get('n_bathrooms') or 1),
            checks, int(f.get('n_safety', 0)), int(f.get('n_amenities', 20))]


class SessionMemory:
    def __init__(self): self.cases = []
    def recall(self, f, n=3):
        return [c for c in self.cases if c.get('neighbourhood') == f.get('neighbourhood') and c.get('room_type') == f.get('room_type')][-n:]
    def save(self, file):
        f, s = file.data['features'], file.data['strategy']
        self.cases.append(dict(neighbourhood=f.get('neighbourhood'), room_type=f.get('room_type'), guests=f.get('accommodates'), target=s.get('target'), price=s.get('recommended_price')))
MEMORY = SessionMemory()
