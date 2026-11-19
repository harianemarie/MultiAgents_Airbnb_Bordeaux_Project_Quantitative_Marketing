# Multi-agent marketing adviser for Airbnb concierge services in Bordeaux

Master 2 project · Quantitative Marketing & Applications · Topic 4.

**Team:** GARANGO Bintou Daouda · TOGNIBO Hariane · KOUAZOUNDE Fifamè Aubierge

When a concierge service takes on a new property, it must analyse the market, choose a target and a price, then write the listing. This project connects these three decisions in **one workflow**, controlled by the property manager.

Five specialised agents share **one property file**. The manager approves the strategy, then the final result.

| Agent | Role | Output |
|---|---|---|
| **Planner** | Reads the request, extracts the property features, coordinates the agents | Property profile, log |
| **Analyst** | Calls the data tools: comparables, price model, availability, reviews | Evidence brief with sources |
| **Strategist** | Segmentation, targeting, positioning, price, minimum stay | Strategy based on the evidence |
| **Creative** | Turns the strategy into content | Title, description, campaign, photo directions, social post |
| **Validator** | Programmed checks + Gemini review | Validation, or return to the responsible agent (2 times maximum) |

The Analyst, Strategist, Creative and Validator call **Gemini** (`gemini-3.5-flash-lite`). The Planner works in Python and calls Gemini only to read a free-text description. **Without a Gemini key, each agent switches to programmed rules**: the notebook still runs fully, and the log shows for each step which mode was used (Gemini or rules).

The whole project is in **one notebook**: `Airbnb_MultiAgent_Project.ipynb` (90 cells), which ends with a **Gradio** interface.

**Online application (Streamlit):** https://4yoabnnmh3uujp3nzmxnjn.streamlit.app
It uses the models exported from the notebook. If the app is asleep, click the button to wake it up and wait about one minute.

**Presentation:** `Presentation_finale.pptx`

---

## Quick start

### Requirements

- **Python 3.12** (or newer), with Jupyter: VS Code + Jupyter extension, JupyterLab, or **Google Colab**.
- The **Inside Airbnb Bordeaux** data files in the same folder as the notebook (see [Data](#data)).
- An **internet connection** (libraries, Hugging Face models, Gemini).
- *Optional:* a free **Gemini API key** ([Google AI Studio](https://aistudio.google.com/apikey)). Without a key, the agents use their programmed rules.
- *Recommended:* a **GPU** (Colab → *Runtime* → *Change runtime type* → GPU) for the review analysis. On CPU, the analysis is reduced to 3 reviews per listing.

### Step 1: Get the project

```
git clone https://github.com/kouazmydia-web/Quantitative_Marketing.git
cd Quantitative_Marketing
```

Or, on GitHub: green **Code** button → **Download ZIP**, then unzip the file.

### Step 2: Install the libraries

Cell 3 of the notebook does it:

```
pip install google-genai shap gradio sentence-transformers sentencepiece protobuf
```

The usual libraries are also needed (already installed on Colab):

```
pip install numpy pandas matplotlib statsmodels scikit-learn plotly torch transformers requests pillow
```

### Step 3: Set up Gemini (optional)

| Environment | Procedure |
|---|---|
| **Google Colab** | **Secrets** panel (key icon, left) → add `GEMINI_API_KEY` and allow notebook access. |
| **Windows (PowerShell)** | `$env:GEMINI_API_KEY="your_key"`, then start VS Code / Jupyter **from the same terminal**. |
| **macOS / Linux** | `export GEMINI_API_KEY="your_key"`, then start Jupyter from this terminal. |

To force the mode without Gemini: `USE_LLM = False` in the settings cell (section 1).

> **Never put the key in the notebook or on GitHub.** The code hides the key in error messages.

### Step 4: Run

Open the notebook and click **Run All**. The first run takes several minutes: hyperparameter search (grid search) and NLP analysis of the reviews. The NLP results are then cached in `nlp_reviews_cache_v2.csv`, so the next runs are faster.

The last cell launches the **Gradio interface** and shows a local link, and a temporary public link on Colab.

---

## Data

Source: **[Inside Airbnb](https://insideairbnb.com/get-the-data/)**, Bordeaux, snapshot of **17 September 2026** (CC BY 4.0 licence).

| File | Content | Used |
|---|---|---|
| `listings (1).csv.gz` | **Detailed** listings (12,063 listings, 90 columns) | Yes |
| `calendar.csv` / `calendar.csv.gz` | Day-by-day availability over one year | Yes |
| `reviews.csv` | Guest reviews, **with the `comments` column** | Yes |
| `neighbourhoods.csv` | List of neighbourhoods | No |

**How the notebook chooses the files.** The `read_file` function takes, in alphabetical order, the first file whose name starts with `listings`, `calendar` or `reviews` and ends with `.csv` or `.gz`. With the current names, the detailed file `listings (1).csv.gz` is read. **Warning:** if you rename it `listings.csv.gz`, the summary file `listings.csv` (few columns) will be read first and the notebook will fail. In that case, delete or move `listings.csv`.

**Large files.** GitHub refuses files over 100 MB: keep the `.csv.gz` versions, which pandas reads directly.

---

## Notebook structure

| Section | Content |
|---|---|
| 1. Settings | Gemini, Bordeaux centre, comparable radii (1, 2, 3 km), similarity weights, outlier thresholds |
| 2–3. Loading and exploration | Missing values, listings without a price (36%), price by stay length, calendar, reviews |
| 4. Cleaning | 12,063 → 7,722 listings with a price → **7,246** after removing quotes for 28 nights or more |
| 5. Features | Number of amenities, safety, kitchen type, distance to the centre (computed automatically) |
| 6. Outliers | Price per m², price per guest, Huber regression → **7,031** listings kept |
| 7. Calendar | Availability on given dates, competitive pressure, minimum stay, seasonality |
| 8. Reviews (NLP) | Multilingual encoder `paraphrase-multilingual-MiniLM-L12-v2` + sentiment model `twitter-xlm-roberta-base-sentiment` on 6 themes (calm, cleanliness, location, parking, family, work) |
| 9. Bordeaux market | Median price (€111/night), price by neighbourhood and capacity, multi-listing hosts |
| 10. Literature | Wang & Nicolau (2017), Gibbs et al. (2018), market tools (PriceLabs, Beyond, AirDNA) |
| 11. Price model | OLS, Lasso, random forest, gradient boosting; 80% range by quantile regression; SHAP explanation |
| 12. Images (optional) | CLIP compares the main photo with descriptions (off by default, ≈ 600 MB) |
| 13. The agents | Configuration, shared file, Analyst tools, JSON contracts, Planner, Analyst, Strategist, Creative, Validator |
| 14. Demo | A property never seen by the models (Christmas week 2026), then a property described in plain language |
| 15. Evaluation | 300 test properties, with programmed rules |
| 16. Interface | Gradio application |
| 17. Conclusion | Results, limits, next steps |

---

## Main functions

| Section | Functions / classes | Role |
|---|---|---|
| 2. Loading | `read_file` | Finds and reads the listings, calendar and reviews files |
| 5. Features | `parse_amenities`, `kitchen_type`, `n_bathrooms`, `haversine_km` | Amenities, kitchen type, bathrooms, distance to the centre |
| 6. Outliers | `extract_surface`, `iqr_outlier` | Surface read from the description, detection of extreme values |
| 7. Calendar | `availability`, `minimum_stay` | Availability on given dates, minimum stay |
| 8. Reviews | `keyword_themes`, `nlp_themes` | Review themes with keywords or with the NLP models |
| 11. Price model | `mae_euros`, `price_drivers` | Error in euros, SHAP explanation of a price |
| 13. Analyst tools | `find_comparables`, `predict_price_range`, `check_availability`, `summarise_reviews` | Comparables, price and range, availability, reviews |
| 13. Agents | `SharedPropertyFile`, `Planner`, `Analyst`, `Strategist`, `Creative`, `Validator` | Shared property file and the five agents |
| 13. Gemini | `ask_llm`, `extract_json`, `parse_request` | Gemini call, reading the JSON answer, reading a free-text description |
| 16. Interface | `build_wow_interface`, `run_analysis`, `simulate` | Gradio interface, running the agents, price simulator |

---

## Using the Gradio interface

| Tab | Role |
|---|---|
| **Agents: new property** | Property form (or pre-filled examples) and start of the analysis |
| **Market evidence** | Map of comparables, price range, SHAP drivers, review themes |
| **Strategy and approval** | Proposed target and price, **manager approval** before creation |
| **Listing and campaign** | Listing, campaign, photo directions, post; **final approval** and download |
| **Agents conversation** | Log of the exchanges between agents (Gemini or rules) |
| **Simulator** | Price simulator based on the property features, without any Gemini call |

A revised strategy needs a new approval. The final approval is only possible if the current strategy is approved and the Validator reports no more issues.

---

## Main results

**Price model** (target: log of the price; errors in euros)

| Indicator | Value |
|---|---|
| Chosen model | Random forest (max depth 15, ≈ 1/3 of the features per split) |
| MAE in cross-validation (5 folds) | **€39.3** / night (gradient boosting: €39.8) |
| Final test (1,407 listings) | mean error **€38.0**, median error **€21.0** |
| Coverage of the 80% range | **69%** (needs calibration) |

**Evaluation of the full workflow** (300 test properties, programmed rules, without Gemini)

| Method | Mean error | Median error | Mean error in % |
|---|---:|---:|---:|
| Neighbourhood median | €80.6 | €42.4 | 48.1% |
| Comparables only | €49.0 | €26.2 | 30.6% |
| Model only | €40.9 | €23.8 | 24.4% |
| Combined workflow | €43.4 | €22.3 | 26.4% |

The combined workflow reduces the median error but increases the mean error compared with the model alone. These results do not show that several Gemini agents improve the price.

---

## Troubleshooting

| Message or symptom | Solution |
|---|---|
| `No listings.csv(.gz) file found` | Put the data files **in the same folder as the notebook** and start Jupyter from this folder. |
| `KeyError` on a column (`price`, `amenities`…) | The `listings` file read is the summary version: keep only the detailed version (see [Data](#data)). |
| `KeyError: 'comments'` | `reviews.csv` is the summary version: download `reviews.csv.gz` (detailed) from Inside Airbnb. |
| `No GEMINI_API_KEY found` | Normal without a key: the agents use their rules. Otherwise, see step 3. |
| Gemini quota error (429) | Free limit reached: wait, or set `USE_LLM = False`. The 300-property evaluation does not use Gemini. |
| The review analysis is very slow | Turn on a GPU, or set `USE_NLP_MODELS = False` (keyword method, faster but less reliable). |
| `SHAP is not installed` | `pip install shap`, then run the cell again. |
| `ModuleNotFoundError: gradio` | `pip install gradio`, then restart the kernel. |

---

## Limits

- Prices are **asking prices**, not booked prices; the surface is known for only 31% of listings.
- Cleaning and part of the feature preparation come before the train/test split; tuning and comparison reuse the same folds (no nested cross-validation).
- The 80% range covers only 69% of observed prices.
- The NLP review analysis has no accuracy test on labelled data. Reviews of nearby properties do not prove the qualities of a new property.
- The online Streamlit application uses review themes computed with keywords: the NLP models could not be loaded when its data were exported. The NLP analysis is in the notebook (section 8).
- In the demo, the Analyst used its rules because a missing surface value (NaN) is refused by the Gemini API.
- Manager approvals are **simulated** in the demos. The Validator can miss unsupported claims.
- No independent user evaluation and no proven commercial gain.

---

## Libraries and models

- pandas, NumPy, Matplotlib, Plotly, statsmodels, scikit-learn, SHAP
- Gradio (notebook interface), Streamlit (online application)
- google-genai: Gemini `gemini-3.5-flash-lite`
- Hugging Face: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `cardiffnlp/twitter-xlm-roberta-base-sentiment`, `openai/clip-vit-base-patch32` (optional)

## Sources

- Inside Airbnb, Bordeaux, 17 September 2026, https://insideairbnb.com (CC BY 4.0)
- Wang, D. & Nicolau, J. L. (2017). Price determinants of sharing economy based accommodation rental. *International Journal of Hospitality Management*.
- Gibbs, C. et al. (2018). Pricing in the sharing economy: a hedonic pricing model applied to Airbnb listings. *Journal of Travel & Tourism Marketing*.
