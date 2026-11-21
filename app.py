# -*- coding: utf-8 -*-
"""
NABD  |  نبض
Heart-Attack Risk Analytics Dashboard  (educational project)
Built from: Heart_Attack_Risk_Analysis_Stacking_MetaLearner.ipynb

Run:  streamlit run app.py
Needs: MedicalDataset.xlsx next to this file (or upload it in the app).
Optional: artifacts/preprocessor.joblib + artifacts/best_heart_attack_model.joblib
          (saved by the notebook). If missing, the app re-trains the same
          Logistic-Regression pipeline automatically.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from health_assistant import ask_health_assistant, build_system_prompt, dev_log

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NABD | نبض", page_icon=":material/ecg_heart:",
                   layout="wide", initial_sidebar_state="collapsed")

# ════════════════════════════════════════════════════════════════════
#  1. STATE + TRANSLATION HELPERS
# ════════════════════════════════════════════════════════════════════
st.session_state.setdefault("lang", "en")
LANG = st.session_state["lang"]
DARK = True
AR = LANG == "ar"


def tr(en, ar):
    return ar if AR else en


RED, YEL, GRN = "#ef4444", "#f59e0b", "#22c55e"
CYAN, VIO, BLUE, PINK, TEAL = "#22d3ee", "#8b5cf6", "#3b82f6", "#ec4899", "#14b8a6"
PALETTE = [CYAN, VIO, PINK, YEL, GRN, BLUE, RED, TEAL]
RISK_SCALE = [[0, GRN], [0.5, YEL], [1, RED]]

# ════════════════════════════════════════════════════════════════════
#  2. CSS  (theme variables, RTL, cards, nav pills)
# ════════════════════════════════════════════════════════════════════
VARS_DARK = """--bg:#080d19;--bg2:#0e1628;--card:#111a2e;--card2:#16223b;--bd:rgba(148,163,184,.16);
--tx:#e6edf7;--mu:#94a3b8;--ac:#22d3ee;--ac2:#8b5cf6;--sh:0 10px 30px rgba(0,0,0,.35);"""
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Readex+Pro:wght@500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,1,0&display=block');
:root{{{VARS_DARK}}}
html,body,.stApp,[class*="css"]{{font-family:'Cairo','Inter',sans-serif;color-scheme:dark;}}
.stApp{{background:radial-gradient(1200px 600px at 85% -10%,rgba(139,92,246,.18),transparent 60%),
        radial-gradient(900px 500px at -10% 10%,rgba(34,211,238,.14),transparent 60%),var(--bg);color:var(--tx);
        {"direction:rtl;" if AR else ""}}}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{{display:none!important}}
.block-container{{padding:1.1rem 2rem 5rem 2rem;max-width:1500px}}
.stApp p,.stApp label,.stApp span,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stMarkdown{{color:var(--tx)}}
{"[data-testid='stMarkdownContainer'],label,.stCaption{text-align:right}" if AR else ""}
small,.stCaption,[data-testid="stCaptionContainer"]{{color:var(--mu)!important}}

/* icons */
.mi{{font-family:'Material Symbols Rounded'!important;font-weight:normal;font-style:normal;font-size:1.25em;line-height:1;display:inline-block;
  vertical-align:-.22em;width:1em;height:1em;overflow:hidden;letter-spacing:normal;text-transform:none;white-space:nowrap;direction:ltr;-webkit-font-smoothing:antialiased;
  font-feature-settings:'liga';font-variation-settings:'FILL' 1,'wght' 500,'GRAD' 0,'opsz' 24}}
.kpi-ic .mi{{font-size:1.7rem;color:var(--c)}}
.card-h .mi{{color:var(--ac);font-size:1.3em}}
.blk-h .ic .mi{{color:#fff;font-size:1.35rem}}
.sec-t .mi{{color:var(--ac);font-size:1.3em}}
.ins .mi{{color:var(--ac);font-size:1.3em}}
.chip .mi{{font-size:1.15em;color:var(--ac)}}
.disc .mi{{color:#f59e0b;font-size:1.4em}}
.discbar .mi{{color:#f59e0b}}
.res h2 .mi{{font-size:1.15em;color:var(--rc)}}
.brand-ic{{width:46px;height:46px;min-width:46px;overflow:hidden;border-radius:14px;display:flex;flex:0 0 46px;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--ac),var(--ac2))}}
.brand-ic .mi{{color:#fff!important;font-size:1.7rem;width:1.7rem;height:1.7rem;vertical-align:0}}
[data-testid="stChatMessage"]{{background:var(--card);border:1px solid var(--bd);border-radius:18px;padding:12px 16px}}
[data-testid="stChatInput"]{{border-radius:16px}}
[data-testid="stChatInput"] textarea,[data-testid="stChatInput"]>div{{background:var(--card2)!important;color:var(--tx)!important}}
.hint{{color:var(--mu);font-size:.82rem}}

/* top bar */
.st-key-topbarbox{{padding:14px 18px;min-height:78px;overflow:visible;box-sizing:border-box;border:1px solid var(--bd);border-radius:18px;
  background:linear-gradient(135deg,var(--card),var(--card2));box-shadow:var(--sh);margin-bottom:12px}}
.tb-in{{display:flex;align-items:center;gap:14px;min-height:50px;margin:0;padding:0}}
.st-key-topbarbox [data-testid="stMarkdownContainer"],.st-key-topbarbox [data-testid="stMarkdown"],.st-key-topbarbox [data-testid="stElementContainer"]{{margin:0!important;padding:0!important;overflow:visible!important}}
.st-key-topbarbox [data-testid="stHorizontalBlock"]{{align-items:center}}
[class*="st-key-exp_"]{{position:relative;margin-bottom:10px}}
[class*="st-key-exp_"] .xcard{{margin:0;height:100%;transition:all .25s;cursor:pointer}}
.xtop{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
.xtop .mi:last-child{{font-size:1.2rem;color:var(--mu);transition:all .25s}}
[class*="st-key-exp_"]:hover .xcard{{border-color:var(--ac);transform:translateY(-4px);box-shadow:0 12px 28px rgba(34,211,238,.18)}}
[class*="st-key-exp_"]:hover .xtop .mi:last-child{{color:var(--ac);transform:translate(3px,-3px)}}
[class*="st-key-exp_"] [data-testid="stElementContainer"]:has(button){{position:absolute;inset:0;width:100%!important;height:100%}}
[class*="st-key-exp_"] [data-testid="stButton"],[class*="st-key-exp_"] button{{width:100%;height:100%}}
[class*="st-key-exp_"] button{{opacity:0;cursor:pointer}}
[class*="st-key-msg_"][class*="_rtl"],[class*="st-key-msg_"][class*="_rtl"] :is([data-testid="stMarkdownContainer"],[data-testid="stCaptionContainer"],[data-testid="stAlert"]){{direction:rtl!important;text-align:right!important}}
[class*="st-key-msg_"][class*="_ltr"],[class*="st-key-msg_"][class*="_ltr"] :is([data-testid="stMarkdownContainer"],[data-testid="stCaptionContainer"],[data-testid="stAlert"]){{direction:ltr!important;text-align:left!important}}
[class*="st-key-msg_"] :is(pre,code){{direction:ltr!important;text-align:left!important}}
[data-testid="stChatMessage"]{{overflow:hidden}}
[data-testid="stChatMessageContent"],[class*="st-key-msg_"]{{min-width:0;max-width:100%}}
[class*="st-key-msg_"] [data-testid="stMarkdownContainer"]{{overflow-wrap:anywhere;word-break:break-word}}
[class*="st-key-msg_"] :is(h1,h2,h3,h4){{font-size:1.12rem!important;font-weight:800;line-height:1.5;padding:.15rem 0 0;margin:.7rem 0 .3rem;text-align:inherit}}
[class*="st-key-msg_"] :is(ul,ol){{margin:.3rem 0 .6rem;list-style-position:outside}}
[class*="st-key-msg_"][class*="_rtl"] :is(ul,ol){{padding-left:0!important;padding-right:1.5rem!important;margin-left:0!important}}
[class*="st-key-msg_"][class*="_ltr"] :is(ul,ol){{padding-right:0!important;padding-left:1.5rem!important}}
[class*="st-key-msg_"] li{{margin-inline-start:0!important;padding-inline-start:.25rem;text-align:start}}
[class*="st-key-msg_"][class*="_rtl"] blockquote{{border-left:none!important;border-right:3px solid var(--bd);padding:.1rem .9rem .1rem 0;margin:.5rem 0}}
.tips{{position:relative;flex:1;height:46px;margin-inline-start:28px;min-width:0;overflow:hidden;direction:ltr;
  -webkit-mask-image:linear-gradient(to right,transparent,#000 5%,#000 95%,transparent);
  mask-image:linear-gradient(to right,transparent,#000 5%,#000 95%,transparent)}}
.tips-track{{display:flex;align-items:center;width:max-content;height:100%;
  animation:tipscroll var(--tipdur) linear infinite;animation-direction:var(--tipdir)}}
.tips:hover .tips-track{{animation-play-state:paused}}
.tip{{display:flex;align-items:center;gap:10px;padding:0 16px;margin-right:14px;height:38px;border-radius:999px;flex:0 0 auto;
  border:1px solid var(--bd);background:rgba(34,211,238,.06);color:var(--tx);font-size:.92rem;font-weight:600;white-space:nowrap}}
.tip .mi{{color:var(--ac);font-size:1.35rem;flex:0 0 auto}}
@keyframes tipscroll{{from{{transform:translateX(0)}}to{{transform:translateX(-50%)}}}}
@media (max-width:820px){{.tips{{display:none}}}}
.brand-sub{{line-height:1.3}}
.brand-wrap{{display:flex;flex-direction:column;align-items:flex-start;direction:ltr}}
.brand{{font-family:'Readex Pro','Cairo',sans-serif!important;font-weight:700!important;font-size:2rem!important;line-height:1.1;
  display:flex;align-items:center;gap:12px}}
.brand .brand-lat{{letter-spacing:.14em}}
.brand-ar{{font-family:'Readex Pro','Cairo',sans-serif!important;font-weight:700;letter-spacing:0;font-size:inherit;line-height:inherit}}
.brand-wave{{width:100%;min-width:150px;height:22px;margin:0 0 2px;overflow:visible}}
.brand-wave path{{fill:none;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke}}
.brand-wave .w0{{stroke:var(--ac);stroke-width:1.5px;opacity:.28}}
.brand-wave .w1{{stroke:var(--ac);stroke-width:2.6px;stroke-dasharray:120 280;filter:drop-shadow(0 0 4px var(--ac));animation:nabdwalk 2.6s linear infinite}}
@keyframes nabdwalk{{from{{stroke-dashoffset:120}}to{{stroke-dashoffset:-280}}}}
.st-key-langfab{{position:fixed;bottom:18px;left:18px;z-index:1000;width:auto!important;display:flex;flex-direction:column;gap:8px}}
.st-key-langfab [data-testid="stElementContainer"]{{width:auto!important}}
.st-key-langfab button{{width:46px;height:46px;min-height:0;padding:0;border-radius:50%;box-shadow:var(--sh);
  background:linear-gradient(135deg,var(--ac),var(--ac2));color:#fff;border:none}}
.st-key-langfab button:hover{{color:#fff;transform:scale(1.08)}}
.st-key-tbtns{{direction:ltr}}
.st-key-tbtns button{{min-height:0;height:38px;width:38px;padding:0;border-radius:10px;font-size:.85rem}}
.st-key-tbtns [data-testid="stHorizontalBlock"]{{gap:6px;justify-content:flex-end}}
.st-key-tbtns [data-testid="stColumn"]{{width:auto!important;flex:0 0 auto!important;min-width:0!important}}
.brand{{font-size:1.45rem;font-weight:800;background:linear-gradient(90deg,var(--ac),var(--ac2));-webkit-background-clip:text;
  -webkit-text-fill-color:transparent}}
.brand-sub{{color:var(--mu);font-size:.78rem;margin-top:-4px}}
.beat .mi{{animation:beat 1.2s infinite}}
@keyframes beat{{0%,100%{{transform:scale(1)}}15%{{transform:scale(1.22)}}30%{{transform:scale(1)}}45%{{transform:scale(1.14)}}}}

/* nav pills */
.st-key-mainnav [role="radiogroup"],.st-key-subnav [role="radiogroup"]{{gap:8px;flex-wrap:wrap}}
.st-key-mainnav [role="radiogroup"]{{justify-content:center}}
.st-key-mainnav label,.st-key-subnav label{{background:var(--card);border:1px solid var(--bd);padding:9px 18px;
  border-radius:14px;cursor:pointer;transition:all .25s;box-shadow:var(--sh)}}
.st-key-subnav label{{padding:6px 14px;border-radius:999px;box-shadow:none}}
.st-key-mainnav label:hover,.st-key-subnav label:hover{{transform:translateY(-3px);border-color:var(--ac)}}
.st-key-mainnav label>div:first-child,.st-key-subnav label>div:first-child{{display:none}}
.st-key-mainnav label p,.st-key-subnav label p{{font-weight:700;font-size:.95rem;margin:0}}
.st-key-mainnav label:has(input:checked),.st-key-subnav label:has(input:checked){{
  background:linear-gradient(135deg,var(--ac),var(--ac2));border-color:transparent;box-shadow:0 8px 24px rgba(34,211,238,.35)}}
.st-key-mainnav label:has(input:checked) p,.st-key-subnav label:has(input:checked) p{{color:#fff!important}}

/* small icon buttons */
.stButton>button{{border-radius:12px;border:1px solid var(--bd);background:var(--card);color:var(--tx);font-weight:700;transition:all .2s}}
.stButton>button:hover{{border-color:var(--ac);transform:translateY(-2px);color:var(--ac)}}
.stFormSubmitButton>button,.st-key-gobtn button{{background:linear-gradient(135deg,var(--ac),var(--ac2))!important;color:#fff!important;
  border:none!important;font-size:1.1rem;padding:.7rem 1rem;box-shadow:0 10px 30px rgba(139,92,246,.4)}}

/* KPI rectangles */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:6px 0 18px}}
.kpi{{display:flex;gap:14px;align-items:center;padding:16px 18px;border-radius:18px;border:1px solid var(--bd);
  background:linear-gradient(135deg,var(--card),var(--card2));box-shadow:var(--sh);position:relative;overflow:hidden;
  transition:transform .25s}}
.kpi:hover{{transform:translateY(-4px)}}
.kpi:before{{content:"";position:absolute;inset:0 auto 0 0;width:5px;background:var(--c)}}
{"[dir=rtl] .kpi:before,.stApp .kpi:before{inset:0 0 0 auto}" if AR else ""}
.kpi-ic{{font-size:1.7rem;width:50px;height:50px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  background:color-mix(in srgb,var(--c) 18%,transparent)}}
.kpi-l{{color:var(--mu);font-size:.8rem;font-weight:600}}
.kpi-v{{font-size:1.65rem;font-weight:800;line-height:1.1;color:var(--tx)}}
.kpi-s{{color:var(--mu);font-size:.72rem}}

/* chart cards */
[class*="st-key-card_"]{{background:linear-gradient(160deg,var(--card),var(--card2));border:1px solid var(--bd);
  border-radius:18px;padding:14px 16px 6px;box-shadow:var(--sh);margin-bottom:14px;transition:all .25s}}
[class*="st-key-card_"]:hover{{border-color:color-mix(in srgb,var(--ac) 55%,transparent)}}
.card-h{{font-weight:800;font-size:.98rem;margin-bottom:2px;display:flex;gap:8px;align-items:center}}
.card-h .dot{{width:9px;height:9px;border-radius:50%;background:linear-gradient(135deg,var(--ac),var(--ac2))}}
.sec-t{{font-size:1.25rem;font-weight:800;margin:6px 0 10px}}

/* hero */
.hero{{padding:28px 30px;border-radius:24px;border:1px solid var(--bd);position:relative;overflow:hidden;margin-bottom:16px;
  background:linear-gradient(120deg,rgba(34,211,238,.16),rgba(139,92,246,.18)),var(--card);box-shadow:var(--sh)}}
.hero h1{{margin:0;font-size:2rem;font-weight:800}}
.hero p{{color:var(--mu);margin:.4rem 0 0;max-width:760px;font-size:1rem}}
.ecg{{position:absolute;bottom:6px;left:0;width:100%;height:60px;opacity:.55}}
.ecg path{{stroke-dasharray:900;stroke-dashoffset:900;animation:dash 3.2s linear infinite}}
@keyframes dash{{to{{stroke-dashoffset:0}}}}

/* insight */
.ins{{padding:14px 16px;border-radius:16px;border:1px solid var(--bd);background:var(--card);margin-bottom:10px;font-size:.92rem}}
.ins b{{color:var(--ac)}}

/* result */
.res{{border-radius:24px;padding:26px 28px;border:2px solid var(--rc);position:relative;overflow:hidden;margin-bottom:14px;
  background:linear-gradient(135deg,color-mix(in srgb,var(--rc) 22%,var(--card)),var(--card));
  box-shadow:0 0 50px color-mix(in srgb,var(--rc) 35%,transparent);animation:pop .5s ease}}
@keyframes pop{{from{{transform:scale(.96);opacity:0}}to{{transform:scale(1);opacity:1}}}}
.res h2{{margin:0;font-size:1.8rem;font-weight:800;color:var(--rc)!important}}
.res .big{{font-size:3.4rem;font-weight:800;color:var(--rc)}}
.disc{{border:1px dashed var(--yl);border-radius:16px;padding:12px 16px;background:rgba(245,158,11,.10);font-weight:600;margin:10px 0}}
.discbar{{position:fixed;bottom:0;left:0;right:0;z-index:999;text-align:center;padding:7px 12px;font-size:.78rem;font-weight:600;
  background:var(--bg2);border-top:1px solid var(--bd);color:var(--mu)}}
.chip{{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.75rem;font-weight:700;border:1px solid var(--bd);
  background:var(--card2);color:var(--mu);margin:2px}}

/* inputs */
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],.stNumberInput input,.stTextInput input{{
  background:var(--card2)!important;color:var(--tx)!important;border-color:var(--bd)!important;border-radius:12px!important}}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="menu"],[data-baseweb="popover"] ul{{background:var(--card2)!important;color:var(--tx)!important}}
[data-baseweb="popover"] li{{color:var(--tx)!important}}
[data-testid="stWidgetLabel"] p{{color:var(--tx)!important;font-weight:600}}
[data-testid="stForm"]{{border:none;padding:0}}
[class*="st-key-blk_"]{{background:linear-gradient(160deg,var(--card),var(--card2));border:1px solid var(--bd);border-radius:20px;
  padding:16px 18px 10px;box-shadow:var(--sh);margin-bottom:12px}}
.blk-h{{font-weight:800;font-size:1.05rem;margin-bottom:8px;display:flex;align-items:center;gap:10px}}
.blk-h span.ic{{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;
  background:linear-gradient(135deg,var(--ac),var(--ac2))}}
[data-testid="stDataFrame"]{{border-radius:14px;overflow:hidden}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  3. CONSTANTS FROM THE NOTEBOOK  (same feature groups / pipeline)
# ════════════════════════════════════════════════════════════════════
TARGET = "HeartAttackRisk"
AGE_ORDER = ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", "Age 40 to 44", "Age 45 to 49",
             "Age 50 to 54", "Age 55 to 59", "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
             "Age 80 or older"]
AGE_SHORT = {a: a.replace("Age ", "").replace(" or older", "+").replace(" to ", "–") for a in AGE_ORDER}
SHORT_ORDER = [AGE_SHORT[a] for a in AGE_ORDER]
GH_ORDER = ["Poor", "Fair", "Good", "Very good", "Excellent"]
BMI_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]
CONTINUOUS_COLS = ["HeightInMeters", "WeightInKilograms", "BMI", "AgeMidpoint", "ComorbidityCount", "LifestyleRiskScore"]
BINARY_COLS = ["HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder",
               "HadKidneyDisease", "HadArthritis", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
               "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands",
               "ChestScan", "AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver", "HighRiskLastYear",
               "CovidPos"]
ORDINAL_COLS = ["GeneralHealth", "AgeCategory"]
NOMINAL_COLS = ["Sex", "HadDiabetes", "SmokerStatus", "ECigaretteUsage", "RaceEthnicityCategory",
                "TetanusLast10Tdap", "BMI_Category"]
DROP_COLS = ["PatientID", "State"]
LEAKY = ["ChronicCount", "Has3PlusChronic"]
COMORB = ["HadAngina", "HadStroke", "HadCOPD", "HadAsthma", "HadKidneyDisease", "HadArthritis", "HadDiabetes",
          "HadDepressiveDisorder"]
DISAB = ["DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking",
         "DifficultyDressingBathing", "DifficultyErrands"]
DEFAULTS = {
    "Sex": ["Female", "Male"],
    "HadDiabetes": ["No", "No, pre-diabetes or borderline diabetes", "Yes", "Yes, but only during pregnancy (female)"],
    "SmokerStatus": ["Never smoked", "Former smoker", "Current smoker - now smokes some days",
                     "Current smoker - now smokes every day"],
    "ECigaretteUsage": ["Never used e-cigarettes in my entire life", "Not at all (right now)", "Use them some days",
                        "Use them every day"],
    "RaceEthnicityCategory": ["White only, Non-Hispanic", "Black only, Non-Hispanic", "Hispanic",
                              "Multiracial, Non-Hispanic", "Other race only, Non-Hispanic"],
    "TetanusLast10Tdap": ["No, did not receive any tetanus shot in the past 10 years",
                          "Yes, received tetanus shot but not sure what type", "Yes, received Tdap",
                          "Yes, received tetanus shot, but not Tdap"],
}
# Arabic display names for categorical values (display only – the model always gets the English value)
VAL_AR = {
    "Female": "أنثى", "Male": "ذكر",
    "Poor": "ضعيفة", "Fair": "مقبولة", "Good": "جيدة", "Very good": "جيدة جداً", "Excellent": "ممتازة",
    "Underweight": "نحافة", "Normal": "طبيعي", "Overweight": "وزن زائد", "Obese": "سمنة",
    "No": "لا", "Yes": "نعم", "No, pre-diabetes or borderline diabetes": "لا، مرحلة ما قبل السكري",
    "Yes, but only during pregnancy (female)": "نعم، أثناء الحمل فقط",
    "Never smoked": "لم أدخن أبداً", "Former smoker": "مدخن سابق",
    "Current smoker - now smokes some days": "مدخن حالي - أحياناً",
    "Current smoker - now smokes every day": "مدخن حالي - يومياً",
    "Never used e-cigarettes in my entire life": "لم أستخدم السيجارة الإلكترونية أبداً",
    "Not at all (right now)": "لا أستخدمها حالياً", "Use them some days": "أستخدمها أحياناً",
    "Use them every day": "أستخدمها يومياً",
    "White only, Non-Hispanic": "أبيض (غير لاتيني)", "Black only, Non-Hispanic": "أسود (غير لاتيني)",
    "Hispanic": "لاتيني", "Multiracial, Non-Hispanic": "متعدد الأعراق",
    "Other race only, Non-Hispanic": "عرق آخر",
    "No, did not receive any tetanus shot in the past 10 years": "لا، لم أتلقَّ تطعيم تيتانوس خلال 10 سنوات",
    "Yes, received tetanus shot but not sure what type": "نعم، تطعيم تيتانوس (النوع غير معروف)",
    "Yes, received Tdap": "نعم، Tdap", "Yes, received tetanus shot, but not Tdap": "نعم، تيتانوس (ليس Tdap)",
}


def disp(v):
    if not AR:
        return v
    if v in VAL_AR:
        return VAL_AR[v]
    if isinstance(v, str) and v.startswith("Age "):
        s = v[4:]
        return s.replace(" or older", "+ سنة").replace(" to ", " – ") + (" سنة" if "+" not in s and "older" not in s else "")
    return v


# ════════════════════════════════════════════════════════════════════
#  4. DATA + MODEL ARTIFACTS
# ════════════════════════════════════════════════════════════════════
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.factor * iqr, q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            X[c] = X[c].clip(lower=self.lower_[c], upper=self.upper_[c])
        return X.values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.log1p(np.asarray(X))

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features)


def build_preprocessor():
    cont = Pipeline([("imputer", SimpleImputer(strategy="median")), ("outlier_clip", OutlierClipper(1.5)),
                     ("log", LogTransformer()), ("scaler", RobustScaler())])
    binp = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    ordp = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("encoder", OrdinalEncoder(categories=[GH_ORDER, AGE_ORDER], handle_unknown="use_encoded_value",
                                                unknown_value=-1))])
    nomp = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))])
    return ColumnTransformer([("cont", cont, CONTINUOUS_COLS), ("binary", binp, BINARY_COLS),
                              ("ordinal", ordp, ORDINAL_COLS), ("nominal", nomp, NOMINAL_COLS)],
                             remainder="drop", verbose_feature_names_out=False)


def engineer(d):
    """Same target-independent feature engineering as the notebook."""
    d = d.copy()
    b = d["AgeCategory"].astype(str).str.extract(r"Age (\d+) to (\d+)")
    d["AgeMidpoint"] = ((b[0].astype(float) + b[1].astype(float)) / 2).fillna(82.5)

    def to_bin(col):
        if col.dtype == "object":
            v = col.astype(str).str.strip().str.lower()
            return v.map({"yes": 1, "no": 0}).fillna(v.str.startswith("yes").astype(int))
        return pd.to_numeric(col, errors="coerce").fillna(0)

    d["ComorbidityCount"] = d[COMORB].apply(to_bin).fillna(0).sum(axis=1)
    smk = (~d["SmokerStatus"].astype(str).str.lower().str.contains("never")).astype(int)
    vap = (~d["ECigaretteUsage"].astype(str).str.lower().str.contains("never")).astype(int)
    d["LifestyleRiskScore"] = smk + vap + (d["BMI"].fillna(d["BMI"].median()) >= 30).astype(int)
    d["BMI_Category"] = pd.cut(d["BMI"], [-np.inf, 18.5, 25, 30, np.inf], labels=BMI_ORDER, right=False).astype(str)
    return d


@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame):
    d = raw.copy()
    if TARGET not in d.columns and "HadHeartAttack" in d.columns:
        d = d.rename(columns={"HadHeartAttack": TARGET})
    d = engineer(d)
    d["Age"] = pd.Categorical(d["AgeCategory"].map(AGE_SHORT), categories=SHORT_ORDER, ordered=True)
    d["HasDiabetes"] = d["HadDiabetes"].isin(["Yes", "Yes, but only during pregnancy (female)"]).astype(int)
    d["CurrentSmoker"] = d["SmokerStatus"].str.contains("Current").astype(int)
    chronic = [TARGET, "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder",
               "HadKidneyDisease", "HadArthritis", "HasDiabetes"]
    d["ChronicCount"] = d[chronic].sum(axis=1)
    d["Has3PlusChronic"] = (d["ChronicCount"] >= 3).astype(int)
    d["HealthScore"] = d["GeneralHealth"].map({g: i + 1 for i, g in enumerate(GH_ORDER)})
    d["DisabCount"] = d[DISAB].sum(axis=1)
    return d


def find_file(names, dirs=(".", "data", "/content", "artifacts")):
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    for dr in dirs:
        for n in names:
            for root in (dr, os.path.join(base, dr)):
                p = os.path.join(root, n)
                if os.path.exists(p):
                    return p
    return None


@st.cache_data(show_spinner=False)
def read_data(path):
    return pd.read_excel(path) if path.lower().endswith(("xlsx", "xls")) else pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def get_artifacts(_d, sig):
    """Load notebook artifacts if possible, otherwise re-train the same LR pipeline."""
    X = _d.drop(columns=[c for c in DROP_COLS + [TARGET] + LEAKY if c in _d.columns])
    y = _d[TARGET]
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=0.1765, stratify=ytmp, random_state=42)
    pre_p = find_file(["preprocessor.joblib"])
    mdl_p = find_file(["best_heart_attack_model.joblib"])
    if pre_p and mdl_p:
        try:
            pre = joblib.load(pre_p)
            bundle = joblib.load(mdl_p)
            pre.transform(Xte.head(5))
            return dict(pre=pre, model=bundle["model"], thr=float(bundle["chosen_threshold"]),
                        name=bundle.get("best_model_name", "Model"), source="notebook",
                        thr_f1=float(bundle.get("threshold_f1", 0.5)), Xte=Xte, yte=yte.values)
        except Exception:
            pass
    pre = build_preprocessor().fit(Xtr)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    lr = LogisticRegression(C=10, class_weight={0: 1, 1: spw}, max_iter=4000).fit(pre.transform(Xtr), ytr)
    pv = lr.predict_proba(pre.transform(Xva))[:, 1]
    p, r, t = precision_recall_curve(yva, pv)
    ok = np.where(r[:-1] >= 0.80)[0]
    thr = float(t[ok].max()) if len(ok) else 0.5
    return dict(pre=pre, model=lr, thr=thr, name="Logistic Regression", source="retrained", thr_f1=0.5,
                Xte=Xte, yte=yte.values)


@st.cache_data(show_spinner=False)
def processed_all(_art, _d, sig):
    X = _d.drop(columns=[c for c in DROP_COLS + [TARGET] + LEAKY if c in _d.columns])
    return np.asarray(_art["pre"].transform(X), dtype=float)


# ════════════════════════════════════════════════════════════════════
#  5. UI HELPERS
# ════════════════════════════════════════════════════════════════════
def mi(name, size=None, color=None):
    sty = ""
    if size:
        sty += f"font-size:{size}px;"
    if color:
        sty += f"color:{color};"
    return '<span class="mi"' + (f' style="{sty}"' if sty else "") + f">{name}</span>"


def show(fig, key, h=400):
    dark = DARK
    fg = "#e6edf7" if dark else "#0f172a"
    grid = "rgba(148,163,184,.16)"
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Cairo, Inter, sans-serif", color=fg),
                      height=h, margin=dict(l=8, r=8, t=24, b=8), colorway=PALETTE,
                      legend=dict(orientation="h", y=-0.22, title=None), title=None)
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def kpis(items):
    html = '<div class="kpi-grid">'
    for ic, lab, val, sub, col in items:
        html += (f'<div class="kpi" style="--c:{col}"><div class="kpi-ic">{mi(ic)}</div><div>'
                 f'<div class="kpi-l">{lab}</div><div class="kpi-v">{val}</div><div class="kpi-s">{sub}</div></div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def card_header(title, icon="insights"):
    st.markdown(f'<div class="card-h">{mi(icon)} {title}</div>', unsafe_allow_html=True)


def disclaimer_box():
    st.markdown(
        '<div class="disc" style="--yl:#f59e0b">' + mi("health_and_safety") + ' ' +
        tr("This tool does NOT replace a doctor. It is only an educational project built on a sample dataset. "
           "For any health concern, please consult a qualified physician.",
           "هذه الأداة لا تغني عن الطبيب، وهي مجرد مشروع تعليمي مبني على بيانات عينة. "
           "في أي استفسار صحي يرجى استشارة طبيب مختص.") + "</div>", unsafe_allow_html=True)


def pct(x, nd=1):
    return f"{x:.{nd}f}%"


# ════════════════════════════════════════════════════════════════════
#  6. CHART REGISTRY  (all notebook EDA figures, grouped by category)
# ════════════════════════════════════════════════════════════════════
CATS = [
    ("pop", "groups", "Population", "السكان"),
    ("body", "monitor_weight", "Body & BMI", "الوزن والـBMI"),
    ("heart", "favorite", "Heart & Stroke", "القلب والسكتة"),
    ("chronic", "stethoscope", "Chronic Conditions", "الأمراض المزمنة"),
    ("life", "smoking_rooms", "Lifestyle", "نمط الحياة"),
    ("func", "accessible", "Function & Mobility", "الحركة والوظائف"),
    ("geo", "map", "Geography", "الجغرافيا"),
    ("corr", "hub", "Correlations & Risk", "الارتباطات والمخاطر"),
]
CHARTS = []


def chart(cat, en, ar, wide=False, h=400):
    def deco(f):
        CHARTS.append(dict(cat=cat, en=en, ar=ar, f=f, wide=wide, h=h, id=f.__name__))
        return f
    return deco


def rate(d, by, col=TARGET):
    return d.groupby(by, observed=True)[col].mean().mul(100).reset_index()


def line_age(d, col, hue, ylab):
    g = d.groupby(["Age", hue], observed=True)[col].mean().mul(100).reset_index()
    f = px.line(g, x="Age", y=col, color=hue, markers=True, category_orders={"Age": SHORT_ORDER},
                labels={col: ylab, "Age": tr("Age", "العمر")})
    f.update_traces(line=dict(width=3), marker=dict(size=8))
    return f


def bar_grouped(d, x, col, hue, ylab, order=None, text=True):
    g = d.groupby([x, hue], observed=True)[col].mean().mul(100).reset_index()
    f = px.bar(g, x=x, y=col, color=hue, barmode="group", labels={col: ylab},
               category_orders={x: order} if order else None, text=g[col].round(1) if text else None)
    f.update_traces(textposition="outside")
    return f


def hbar(s, scale=RISK_SCALE, fmt="{:.1f}"):
    s = s.sort_values()
    f = px.bar(x=s.values, y=s.index, orientation="h", color=s.values, color_continuous_scale=scale,
               text=[fmt.format(v) for v in s.values], labels={"x": "%", "y": ""})
    f.update_layout(coloraxis_showscale=False)
    f.update_traces(textposition="outside")
    return f


def heat(z, x, y, scale=None, fmt=".1f", zmin=None, zmax=None):
    f = px.imshow(z, x=x, y=y, aspect="auto", color_continuous_scale=scale or RISK_SCALE, text_auto=fmt, zmin=zmin,
                  zmax=zmax)
    f.update_layout(coloraxis_showscale=False)
    return f


# ---- Population -------------------------------------------------------------
CONDS_ALL = [TARGET, "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder",
             "HadKidneyDisease", "HadArthritis", "DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyWalking",
             "DifficultyConcentrating", "DifficultyErrands", "DifficultyDressingBathing", "CovidPos", "AlcoholDrinkers",
             "FluVaxLast12", "PneumoVaxEver", "HIVTesting"]


@chart("pop", "Prevalence of all conditions & behaviours (%)", "انتشار كل الحالات والسلوكيات (%)", wide=True, h=620)
def prev_all(d):
    return hbar(d[[c for c in CONDS_ALL if c in d]].mean().mul(100), scale="Tealgrn")


@chart("pop", "Age distribution", "توزيع الفئات العمرية")
def age_dist(d):
    s = d["Age"].value_counts().reindex(SHORT_ORDER)
    return px.bar(x=s.index, y=s.values, color=s.values, color_continuous_scale="Tealgrn",
                  labels={"x": tr("Age", "العمر"), "y": tr("Patients", "المرضى")}).update_layout(coloraxis_showscale=False)


@chart("pop", "Sex distribution", "توزيع الجنس")
def sex_pie(d):
    s = d["Sex"].value_counts()
    f = px.pie(names=[disp(i) for i in s.index], values=s.values, hole=.55, color_discrete_sequence=[PINK, BLUE])
    f.update_traces(textinfo="percent+label")
    return f


@chart("pop", "General health status", "الحالة الصحية العامة")
def gh_dist(d):
    s = d["GeneralHealth"].value_counts().reindex(GH_ORDER)
    return px.bar(x=[disp(i) for i in s.index], y=s.values, color=s.index, text=s.values,
                  color_discrete_sequence=[RED, "#fb923c", YEL, "#84cc16", GRN]).update_layout(showlegend=False)


@chart("pop", "General health composition by age (%)", "تركيب الحالة الصحية العامة حسب العمر (%)", wide=True)
def gh_age(d):
    g = pd.crosstab(d["Age"], d["GeneralHealth"], normalize="index").mul(100)[GH_ORDER].reset_index().melt("Age")
    return px.bar(g, x="Age", y="value", color="GeneralHealth", category_orders={"GeneralHealth": GH_ORDER, "Age": SHORT_ORDER},
                  color_discrete_sequence=[RED, "#fb923c", YEL, "#84cc16", GRN], labels={"value": "%"})


@chart("pop", "Key conditions by race / ethnicity (%)", "أهم الحالات حسب العرق (%)", wide=True)
def race_cond(d):
    cols = [TARGET, "HasDiabetes", "HadStroke", "HadCOPD", "HadDepressiveDisorder"]
    g = d.groupby("RaceEthnicityCategory")[cols].mean().mul(100).reset_index().melt("RaceEthnicityCategory")
    return px.bar(g, x="RaceEthnicityCategory", y="value", color="variable", barmode="group", labels={"value": "%", "RaceEthnicityCategory": ""})


@chart("pop", "Preventive care & testing rates (%)", "معدلات الرعاية الوقائية والفحوصات (%)")
def prevent(d):
    s = d[["FluVaxLast12", "PneumoVaxEver", "HIVTesting", "ChestScan"]].mean().mul(100)
    return hbar(s, scale="Tealgrn")


@chart("pop", "COVID-19 positive rate by age & sex (%)", "نسبة كوفيد الإيجابية حسب العمر والجنس (%)")
def covid(d):
    return line_age(d, "CovidPos", "Sex", "%")


# ---- Body ------------------------------------------------------------------
@chart("body", "BMI distribution by category", "توزيع الـBMI حسب الفئة")
def bmi_hist(d):
    return px.histogram(d, x="BMI", nbins=60, color="BMI_Category", category_orders={"BMI_Category": BMI_ORDER},
                        color_discrete_sequence=[BLUE, GRN, YEL, RED])


@chart("body", "BMI category share (%)", "نسبة فئات الـBMI (%)")
def bmi_share(d):
    s = d["BMI_Category"].value_counts(normalize=True).reindex(BMI_ORDER).mul(100)
    return px.bar(x=s.index, y=s.values, color=s.index, color_discrete_sequence=[BLUE, GRN, YEL, RED],
                  text=[f"{v:.1f}%" for v in s.values]).update_layout(showlegend=False)


@chart("body", "Heart attack by BMI category & sex (%)", "النوبة القلبية حسب الـBMI والجنس (%)")
def ha_bmi(d):
    return bar_grouped(d, "BMI_Category", TARGET, "Sex", "%", BMI_ORDER)


@chart("body", "Stroke by BMI category & sex (%)", "السكتة حسب الـBMI والجنس (%)")
def st_bmi(d):
    return bar_grouped(d, "BMI_Category", "HadStroke", "Sex", "%", BMI_ORDER)


@chart("body", "Diabetes status by BMI category (%)", "حالة السكري حسب فئة الـBMI (%)")
def diab_bmi(d):
    g = pd.crosstab(d["BMI_Category"], d["HadDiabetes"], normalize="index").mul(100).reindex(BMI_ORDER).reset_index().melt("BMI_Category")
    return px.bar(g, x="BMI_Category", y="value", color="HadDiabetes", labels={"value": "%"})


@chart("body", "Diabetes heatmap: age × BMI (%)", "خريطة السكري: العمر × الـBMI (%)")
def diab_heat(d):
    p = d.pivot_table(index="BMI_Category", columns="Age", values="HasDiabetes", aggfunc="mean", observed=True).mul(100).reindex(BMI_ORDER)
    return heat(p.values, list(p.columns), list(p.index), fmt=".0f")


@chart("body", "Depression by BMI category (%)", "الاكتئاب حسب الـBMI (%)")
def dep_bmi(d):
    s = rate(d, "BMI_Category", "HadDepressiveDisorder").set_index("BMI_Category").reindex(BMI_ORDER)["HadDepressiveDisorder"]
    return px.bar(x=s.index, y=s.values, color=s.values, color_continuous_scale=RISK_SCALE, text=[f"{v:.1f}%" for v in s.values]).update_layout(coloraxis_showscale=False)


@chart("body", "BMI by general health & sex", "الـBMI حسب الحالة العامة والجنس")
def bmi_box(d):
    return px.box(d, x="GeneralHealth", y="BMI", color="Sex", category_orders={"GeneralHealth": GH_ORDER})


# ---- Heart ----------------------------------------------------------------
@chart("heart", "Heart attack by age & sex (%)", "النوبة القلبية حسب العمر والجنس (%)", wide=True)
def ha_age(d):
    return line_age(d, TARGET, "Sex", "%")


@chart("heart", "Stroke by age & sex (%)", "السكتة حسب العمر والجنس (%)")
def stroke_age(d):
    return line_age(d, "HadStroke", "Sex", "%")


@chart("heart", "Heart conditions & COPD by smoking status (%)", "أمراض القلب وCOPD حسب التدخين (%)")
def heart_smoke(d):
    g = d.groupby("SmokerStatus")[[TARGET, "HadAngina", "HadStroke", "HadCOPD"]].mean().mul(100).reset_index().melt("SmokerStatus")
    f = px.bar(g, x="SmokerStatus", y="value", color="variable", barmode="group", labels={"value": "%", "SmokerStatus": ""})
    f.update_xaxes(showticklabels=False)
    return f


@chart("heart", "Top 20 highest-risk groups (age + sex + smoking + BMI)", "أعلى 20 مجموعة خطورة (عمر + جنس + تدخين + BMI)", wide=True, h=620)
def top_combo(d):
    t = d.copy()
    t["grp"] = (t["Age"].astype(str) + " | " + t["Sex"] + " | " + t["SmokerStatus"].str.split(" - ").str[0] + " | " + t["BMI_Category"])
    g = t.groupby("grp")[TARGET].agg(["mean", "size"])
    g = g[g["size"] >= 15].sort_values("mean", ascending=False).head(20)
    s = (g["mean"] * 100)
    return hbar(s)


@chart("heart", "Relative risk vs. baseline (never smoked + normal BMI = 1.0)", "الخطر النسبي مقابل الأساس (غير مدخن + BMI طبيعي = 1)")
def rel_risk(d):
    p = d.pivot_table(index="SmokerStatus", columns="BMI_Category", values=TARGET, aggfunc="mean", observed=True)
    base = p.loc[p.index.str.contains("Never"), "Normal"].iloc[0] if "Normal" in p else np.nan
    p = (p / base).reindex(columns=[c for c in BMI_ORDER if c in p.columns])
    return heat(p.values, list(p.columns), [i[:28] for i in p.index], fmt=".2f")


@chart("heart", "Detailed smoking impact on cardio-respiratory conditions (%)", "تأثير التدخين التفصيلي على القلب والتنفس (%)", wide=True)
def smoke_detail(d):
    cols = [TARGET, "HadAngina", "HadStroke", "HadCOPD", "HadAsthma"]
    p = d.groupby("SmokerStatus")[cols].mean().mul(100)
    return heat(p.values, cols, [i[:34] for i in p.index], fmt=".1f")


# ---- Chronic --------------------------------------------------------------
@chart("chronic", "Number of chronic conditions per person", "عدد الأمراض المزمنة لكل شخص")
def chronic_cnt(d):
    s = d["ChronicCount"].value_counts().sort_index()
    return px.bar(x=s.index.astype(str), y=s.values, color=s.index, color_continuous_scale=RISK_SCALE).update_layout(coloraxis_showscale=False)


@chart("chronic", "Average chronic conditions by age", "متوسط الأمراض المزمنة حسب العمر")
def chronic_age(d):
    g = d.groupby("Age", observed=True)["ChronicCount"].mean().reset_index()
    return px.area(g, x="Age", y="ChronicCount", markers=True).update_traces(line_color=CYAN)


@chart("chronic", "P(≥3 chronic conditions) by age & sex (%)", "احتمال ≥3 أمراض مزمنة حسب العمر والجنس (%)")
def chronic3(d):
    return line_age(d, "Has3PlusChronic", "Sex", "%")


@chart("chronic", "Comorbidity matrix: P(B | A) %", "مصفوفة الأمراض المصاحبة: P(B | A) %", wide=True, h=520)
def comorb(d):
    cols = [TARGET, "HadAngina", "HadStroke", "HadCOPD", "HasDiabetes", "HadKidneyDisease", "HadArthritis", "HadDepressiveDisorder", "HadAsthma"]
    m = np.array([[(d[a] & d[b]).sum() / max(d[a].sum(), 1) * 100 for b in cols] for a in cols])
    return heat(m, cols, cols, fmt=".0f")


@chart("chronic", "How depression affects other conditions (%)", "تأثير الاكتئاب على الحالات الأخرى (%)")
def dep_impact(d):
    cols = [TARGET, "HadAngina", "HadStroke", "HadCOPD", "HasDiabetes", "HadArthritis", "DifficultyWalking"]
    g = d.groupby("HadDepressiveDisorder")[cols].mean().mul(100).rename(index={0: tr("No depression", "بدون اكتئاب"), 1: tr("Depression", "اكتئاب")})
    g = g.reset_index().melt("HadDepressiveDisorder")
    return px.bar(g, x="variable", y="value", color="HadDepressiveDisorder", barmode="group", labels={"value": "%", "variable": ""})


@chart("chronic", "Kidney disease by age & diabetes status (%)", "أمراض الكلى حسب العمر والسكري (%)")
def kidney(d):
    t = d.assign(Diabetes=d["HasDiabetes"].map({0: tr("No diabetes", "بدون سكري"), 1: tr("Diabetes", "سكري")}))
    return line_age(t, "HadKidneyDisease", "Diabetes", "%")


@chart("chronic", "Arthritis by age & sex (%)", "التهاب المفاصل حسب العمر والجنس (%)")
def arthritis(d):
    return line_age(d, "HadArthritis", "Sex", "%")


@chart("chronic", "Diabetes by age & sex (%)", "السكري حسب العمر والجنس (%)")
def diab_age_sex(d):
    return line_age(d, "HasDiabetes", "Sex", "%")


@chart("chronic", "Depression by age & sex (%)", "الاكتئاب حسب العمر والجنس (%)")
def dep_age(d):
    return line_age(d, "HadDepressiveDisorder", "Sex", "%")


@chart("chronic", "COPD by age & current smoking (%)", "COPD حسب العمر والتدخين الحالي (%)")
def copd_age(d):
    t = d.assign(Smoker=d["CurrentSmoker"].map({0: tr("Non-current", "غير مدخن حالياً"), 1: tr("Current smoker", "مدخن حالي")}))
    return line_age(t, "HadCOPD", "Smoker", "%")


# ---- Lifestyle ------------------------------------------------------------
@chart("life", "Smoking status distribution", "توزيع حالة التدخين")
def smoke_dist(d):
    s = d["SmokerStatus"].value_counts()
    f = px.pie(names=[i[:34] for i in s.index], values=s.values, hole=.55, color_discrete_sequence=[GRN, YEL, "#fb923c", RED])
    f.update_traces(textinfo="percent")
    return f


@chart("life", "COPD by smoking × e-cigarette use (%)", "COPD حسب التدخين × السيجارة الإلكترونية (%)")
def copd_ecig(d):
    p = d.pivot_table(index="SmokerStatus", columns="ECigaretteUsage", values="HadCOPD", aggfunc="mean").mul(100)
    return heat(p.values, [c[:22] for c in p.columns], [i[:30] for i in p.index], fmt=".1f")


@chart("life", "Alcohol drinkers by age & sex (%)", "شرب الكحول حسب العمر والجنس (%)")
def alcohol(d):
    return line_age(d, "AlcoholDrinkers", "Sex", "%")


@chart("life", "High-risk behaviour (last year) by age & sex (%)", "السلوك عالي الخطورة حسب العمر والجنس (%)")
def highrisk(d):
    return line_age(d, "HighRiskLastYear", "Sex", "%")


# ---- Function -------------------------------------------------------------
@chart("func", "Difficulty walking by age & arthritis (%)", "صعوبة المشي حسب العمر والتهاب المفاصل (%)")
def walk_arth(d):
    t = d.assign(Arthritis=d["HadArthritis"].map({0: tr("No arthritis", "بدون التهاب"), 1: tr("Arthritis", "التهاب مفاصل")}))
    return line_age(t, "DifficultyWalking", "Arthritis", "%")


@chart("func", "Difficulty walking by age & sex (%)", "صعوبة المشي حسب العمر والجنس (%)")
def walk_sex(d):
    return line_age(d, "DifficultyWalking", "Sex", "%")


@chart("func", "Prevalence of functional disabilities (%)", "انتشار الإعاقات الوظيفية (%)")
def disab_prev(d):
    return hbar(d[DISAB].mean().mul(100), scale="Purp")


@chart("func", "Average number of disabilities by age", "متوسط عدد الإعاقات حسب العمر")
def disab_age(d):
    g = d.groupby("Age", observed=True)["DisabCount"].mean().reset_index()
    return px.bar(g, x="Age", y="DisabCount", color="DisabCount", color_continuous_scale=RISK_SCALE).update_layout(coloraxis_showscale=False)


@chart("func", "Functional decline cascade by age (%)", "تسلسل التدهور الوظيفي حسب العمر (%)", wide=True)
def cascade(d):
    g = d.groupby("Age", observed=True)[DISAB].mean().mul(100).reset_index().melt("Age")
    f = px.line(g, x="Age", y="value", color="variable", markers=True, labels={"value": "%"})
    f.update_traces(line=dict(width=3))
    return f


# ---- Geography ------------------------------------------------------------
def top_states(d, col, n=10):
    return hbar(d.groupby("State")[col].mean().mul(100).sort_values(ascending=False).head(n))


@chart("geo", "Top 10 states – heart attack (%)", "أعلى 10 ولايات – النوبة القلبية (%)", h=440)
def geo_ha(d):
    return top_states(d, TARGET)


@chart("geo", "Top 10 states – diabetes (%)", "أعلى 10 ولايات – السكري (%)", h=440)
def geo_dia(d):
    return top_states(d, "HasDiabetes")


@chart("geo", "Top 10 states – depression (%)", "أعلى 10 ولايات – الاكتئاب (%)", h=440)
def geo_dep(d):
    return top_states(d, "HadDepressiveDisorder")


@chart("geo", "Top 15 states – current smoking (%)", "أعلى 15 ولاية – التدخين الحالي (%)", h=520)
def geo_smk(d):
    return top_states(d, "CurrentSmoker", 15)


# ---- Correlations -----------------------------------------------------------
@chart("corr", "Correlation heatmap of key indicators", "خريطة الارتباط للمؤشرات الأساسية", wide=True, h=560)
def corr_heat(d):
    cols = ["AgeMidpoint", "BMI", TARGET, "HadAngina", "HadStroke", "HasDiabetes", "CurrentSmoker", "HadCOPD",
            "HadKidneyDisease", "HadArthritis", "HadDepressiveDisorder", "DifficultyWalking", "HealthScore"]
    c = d[cols].corr()
    return px.imshow(c, x=cols, y=cols, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto")


@chart("corr", "Correlation with general-health score (higher = healthier)", "الارتباط مع درجة الصحة العامة (أعلى = أصح)", h=480)
def corr_gh(d):
    cols = [c for c in d.select_dtypes("number").columns if c not in ("HealthScore", "PatientID")]
    s = d[cols].corrwith(d["HealthScore"]).dropna().sort_values()
    s = pd.concat([s.head(8), s.tail(6)])
    f = px.bar(x=s.values, y=s.index, orientation="h", color=s.values, color_continuous_scale=RISK_SCALE[::-1])
    return f.update_layout(coloraxis_showscale=False)


@chart("corr", "Age-standardised prevalence by sex (%)", "الانتشار المعيَّر بالعمر حسب الجنس (%)", h=480)
def age_std(d):
    cols = [TARGET, "HadStroke", "HasDiabetes", "HadCOPD", "HadDepressiveDisorder", "HadArthritis", "HadKidneyDisease"]
    g = d.groupby(["Age", "Sex"], observed=True)[cols].mean().groupby("Sex").mean().mul(100).reset_index().melt("Sex")
    return px.bar(g, x="variable", y="value", color="Sex", barmode="group", labels={"value": "%", "variable": ""})


@chart("corr", "Engineered features: comorbidity burden & lifestyle score", "الميزات المشتقة: عبء الأمراض ودرجة نمط الحياة", h=480)
def eng_feat(d):
    g = d.groupby("LifestyleRiskScore")[TARGET].mean().mul(100).reset_index()
    return px.bar(g, x="LifestyleRiskScore", y=TARGET, color=TARGET, color_continuous_scale=RISK_SCALE,
                  text=g[TARGET].round(1), labels={TARGET: "%"}).update_layout(coloraxis_showscale=False)


# ════════════════════════════════════════════════════════════════════
#  7. PAGES
# ════════════════════════════════════════════════════════════════════
TIPS = [
    ("directions_walk", "Aim for about 150 minutes of moderate activity a week.", "استهدف نحو 150 دقيقة نشاط معتدل أسبوعياً."),
    ("smoking_rooms", "Quitting smoking lowers heart risk at any age.", "الإقلاع عن التدخين يقلل خطر القلب في أي عمر."),
    ("bedtime", "Most adults need 7 to 9 hours of sleep a night.", "معظم البالغين يحتاجون من 7 إلى 9 ساعات نوم ليلاً."),
    ("restaurant", "Go easy on salt, ultra-processed food and sugary drinks.", "قلّل الملح والأطعمة المصنّعة والمشروبات السكرية."),
    ("monitor_heart", "Check your blood pressure regularly. High pressure often has no symptoms.", "قِس ضغطك بانتظام. الضغط المرتفع غالباً بلا أعراض."),
    ("self_improvement", "Short walks and slow breathing help ease everyday stress.", "المشي القصير والتنفس البطيء يخففان توتر الحياة اليومية."),
]


def tips_html():
    one = "".join(f'<div class="tip" style="direction:{"rtl" if AR else "ltr"}">{mi(ic)}<span>{tr(en, ar)}</span></div>'
                  for ic, en, ar in TIPS)
    return (f'<div class="tips" style="--tipdur:{len(TIPS) * 9}s;--tipdir:{"reverse" if AR else "normal"}">'
            f'<div class="tips-track">{one}{one}</div></div>')


def topbar():
    with st.container(key="topbarbox"):
        wave = ('<svg class="brand-wave" viewBox="0 0 180 24" preserveAspectRatio="none">'
                '<path class="w0" pathLength="400" d="M0 13 H34 L41 13 L47 3 L55 22 L61 13 H104 L111 13 L117 2 L126 23 L132 13 H180"/>'
                '<path class="w1" pathLength="400" d="M0 13 H34 L41 13 L47 3 L55 22 L61 13 H104 L111 13 L117 2 L126 23 L132 13 H180"/></svg>')
        st.markdown(f'<div class="tb-in"><span class="brand-ic beat">{mi("ecg_heart")}</span><div class="brand-wrap">'
                    f'<div class="brand"><span class="brand-lat">NABD</span><span class="brand-ar">نبض</span></div>{wave}'
                    f'<div class="brand-sub">{tr("Heart-attack risk analytics", "تحليلات مخاطر النوبة القلبية")}</div></div>'
                    f'{tips_html()}</div>',
                    unsafe_allow_html=True)
    with st.container(key="langfab"):
        st.button("", key="langbtn", icon=":material/translate:",
                  on_click=lambda: st.session_state.update(lang="en" if AR else "ar"),
                  help=tr("Switch to Arabic", "التحويل للإنجليزية"))


PAGES = [("home", "home", "Overview", "نظرة عامة"), ("eda", "monitoring", "Analysis", "التحليل"),
         ("ins", "lightbulb", "Insights", "الرؤى"),
         ("seg", "bubble_chart", "Segments", "تقسيم المرضى"), ("mdl", "model_training", "Models", "النماذج"),
         ("pred", "medical_services", "Predict", "التوقع"),
         ("rec", "checklist", "Recommendations", "التوصيات"),
         ("chat", "smart_toy", "Assistant", "المساعد")]


def nav():
    with st.container(key="mainnav"):
        return st.radio("nav", [p[0] for p in PAGES], horizontal=True, label_visibility="collapsed", key="page",
                        format_func=lambda k: next(f":material/{p[1]}:  {tr(p[2], p[3])}" for p in PAGES if p[0] == k))


def ecg_svg():
    return (f'<svg class="ecg" viewBox="0 0 1200 60" preserveAspectRatio="none"><path d="M0 30 H220 L240 30 L255 8 L275 52 '
            'L292 30 H520 L540 30 L555 4 L578 56 L596 30 H860 L880 30 L895 10 L915 50 L930 30 H1200" fill="none" '
            f'stroke="{CYAN}" stroke-width="2.5"/></svg>')


def page_home(d, art):
    st.markdown(f'<div class="hero"><h1>{tr("Understand heart-attack risk through data", "افهم مخاطر النوبة القلبية من خلال البيانات")}</h1>'
                f'<p>{tr("An interactive dashboard that explores a medical dataset, segments patients, compares machine-learning models and lets you try a risk-prediction demo.", "داشبورد تفاعلي يستكشف بيانات طبية، يقسّم المرضى، يقارن نماذج تعلّم الآلة، ويتيح لك تجربة توقع المخاطر.")}</p>'
                f'{ecg_svg()}</div>', unsafe_allow_html=True)
    kpis([("medical_services", tr("Patients", "عدد المرضى"), f"{len(d):,}", tr("in the dataset", "في البيانات"), CYAN),
          ("favorite", tr("Heart-attack history", "تاريخ نوبة قلبية"), pct(d[TARGET].mean() * 100), f"{int(d[TARGET].sum()):,} {tr('patients', 'مريض')}", RED),
          ("cake", tr("Average age", "متوسط العمر"), f"{d['AgeMidpoint'].mean():.0f}", tr("years (band midpoint)", "سنة (منتصف الفئة)"), VIO),
          ("monitor_weight", tr("Average BMI", "متوسط BMI"), f"{d['BMI'].mean():.1f}", f"{pct((d['BMI'] >= 30).mean() * 100, 0)} {tr('obese', 'سمنة')}", YEL),
          ("water_drop", tr("Diabetes", "السكري"), pct(d["HasDiabetes"].mean() * 100), tr("incl. gestational", "يشمل سكر الحمل"), PINK),
          ("smoking_rooms", tr("Current smokers", "المدخنون حالياً"), pct(d["CurrentSmoker"].mean() * 100), tr("daily or some days", "يومياً أو أحياناً"), GRN)])
    # auto insights
    ag = d.groupby("Age", observed=True)[TARGET].mean().mul(100)
    sm = d.groupby("CurrentSmoker")[TARGET].mean()
    sx = d.groupby("Sex")[TARGET].mean().mul(100)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="ins">{mi("trending_up")} {tr("Highest heart-attack rate is in age group", "أعلى معدل نوبات قلبية في الفئة العمرية")} <b>{ag.idxmax()}</b> ({ag.max():.1f}%).</div>', unsafe_allow_html=True)
    if 0 in sm.index and 1 in sm.index and sm[0] > 0:
        c2.markdown(f'<div class="ins">{mi("smoking_rooms")} {tr("Current smokers have", "المدخنون حالياً لديهم")} <b>{sm[1] / sm[0]:.1f}×</b> {tr("the heart-attack rate of non-smokers in this sample.", "معدل النوبات مقارنة بغير المدخنين في هذه العينة.")}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="ins">{mi("wc")} {tr("Rate by sex:", "المعدل حسب الجنس:")} ' + " · ".join(f"<b>{disp(k)}</b> {v:.1f}%" for k, v in sx.items()) + "</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    for col, fn, ttl in [(a, ha_age, tr("Heart attack by age & sex (%)", "النوبة القلبية حسب العمر والجنس (%)")),
                         (b, prev_all, tr("Prevalence of conditions (%)", "انتشار الحالات (%)"))]:
        with col, st.container(key=f"card_home_{fn.__name__}"):
            card_header(ttl)
            show(fn(d), f"home_{fn.__name__}", 430)
    st.markdown(f'<div class="sec-t">{mi("explore")} {tr("Explore", "استكشف")}</div>', unsafe_allow_html=True)
    desc = {
        "eda": tr(f"{len(CHARTS)} interactive charts in 8 groups", f"{len(CHARTS)} رسم تفاعلي في 8 مجموعات"),
        "ins": tr("Key findings and risk factors", "أهم النتائج وعوامل الخطورة"),
        "seg": tr("PCA · K-Means · GMM · t-SNE · anomalies", "PCA · K-Means · GMM · t-SNE · الشواذ"),
        "mdl": tr("ROC · PR · threshold explorer", "ROC · PR · مستكشف العتبة"),
        "pred": tr("Enter your data, get a colour-coded result", "أدخل بياناتك وشاهد النتيجة الملونة"),
        "rec": tr("Healthy-habit tips based on the data", "نصائح لعادات صحية مبنية على البيانات"),
        "chat": tr("Ask a health question (Gemini)", "اسأل سؤالاً صحياً (Gemini)"),
    }
    targets = [p for p in PAGES if p[0] != "home"]
    for r0 in range(0, len(targets), 4):
        row = st.columns(4)
        for col, (k, ic, en, ar) in zip(row, targets[r0:r0 + 4]):
            with col, st.container(key=f"exp_{k}"):
                st.markdown(f'<div class="ins xcard"><div class="xtop">{mi(ic, 34)}{mi("arrow_outward")}</div>'
                            f'<b>{tr(en, ar)}</b><br><span style="color:var(--mu)">{desc[k]}</span></div>', unsafe_allow_html=True)
                st.button(tr(en, ar), key=f"go_{k}", on_click=lambda k=k: st.session_state.update(page=k))
    disclaimer_box()


def page_eda(d):
    st.markdown(f'<div class="sec-t">{mi("tune")} {tr("Filters", "الفلاتر")}</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.2, 2, 1.6, 1.6])
    sex = f1.multiselect(tr("Sex", "الجنس"), sorted(d["Sex"].unique()), default=sorted(d["Sex"].unique()), format_func=disp)
    ages = f2.select_slider(tr("Age range", "المدى العمري"), options=SHORT_ORDER, value=(SHORT_ORDER[0], SHORT_ORDER[-1]))
    bmi = f3.multiselect(tr("BMI category", "فئة BMI"), BMI_ORDER, default=BMI_ORDER, format_func=disp)
    smk = f4.multiselect(tr("Smoking", "التدخين"), sorted(d["SmokerStatus"].unique()), default=sorted(d["SmokerStatus"].unique()), format_func=lambda x: disp(x))
    lo, hi = SHORT_ORDER.index(ages[0]), SHORT_ORDER.index(ages[1])
    m = d["Sex"].isin(sex) & d["BMI_Category"].isin(bmi) & d["SmokerStatus"].isin(smk) & d["Age"].isin(SHORT_ORDER[lo:hi + 1])
    f = d[m]
    if len(f) < 30:
        st.warning(tr("Too few records after filtering.", "عدد السجلات قليل جداً بعد الفلترة."))
        return
    kpis([("medical_services", tr("Patients", "المرضى"), f"{len(f):,}", pct(len(f) / len(d) * 100, 0) + tr(" of all", " من الكل"), CYAN),
          ("favorite", tr("Heart attack", "نوبة قلبية"), pct(f[TARGET].mean() * 100), f"{int(f[TARGET].sum())} {tr('cases', 'حالة')}", RED),
          ("monitor_weight", tr("Avg BMI", "متوسط BMI"), f"{f['BMI'].mean():.1f}", "", YEL),
          ("water_drop", tr("Diabetes", "السكري"), pct(f["HasDiabetes"].mean() * 100), "", PINK),
          ("smoking_rooms", tr("Smokers", "مدخنون"), pct(f["CurrentSmoker"].mean() * 100), "", GRN),
          ("stethoscope", tr("Avg chronic", "متوسط مزمنة"), f"{f['ChronicCount'].mean():.2f}", tr("conditions / person", "حالة / شخص"), VIO)])
    with st.container(key="subnav"):
        cat = st.radio("cat", [c[0] for c in CATS], horizontal=True, label_visibility="collapsed", key="eda_cat",
                       format_func=lambda k: next(f":material/{c[1]}: {tr(c[2], c[3])}" for c in CATS if c[0] == k))
    catic = next(c[1] for c in CATS if c[0] == cat)
    items = [c for c in CHARTS if c["cat"] == cat]
    i = 0
    while i < len(items):
        it = items[i]
        if it["wide"] or i == len(items) - 1:
            with st.container(key=f"card_{it['id']}"):
                card_header(tr(it["en"], it["ar"]), catic)
                try:
                    show(it["f"](f), f"eda_{it['id']}", it["h"])
                except Exception as e:  # keep dashboard alive
                    st.info(tr("Chart unavailable for the current filter.", "الرسم غير متاح مع الفلتر الحالي.") + f" ({type(e).__name__})")
            i += 1
        else:
            nxt = items[i + 1]
            if nxt["wide"]:
                with st.container(key=f"card_{it['id']}"):
                    card_header(tr(it["en"], it["ar"]), catic)
                    show(it["f"](f), f"eda_{it['id']}", it["h"])
                i += 1
                continue
            cols = st.columns(2)
            for col, c in zip(cols, (it, nxt)):
                with col, st.container(key=f"card_{c['id']}"):
                    card_header(tr(c["en"], c["ar"]), catic)
                    try:
                        show(c["f"](f), f"eda_{c['id']}", max(it["h"], nxt["h"]))
                    except Exception as e:
                        st.info(tr("Chart unavailable for the current filter.", "الرسم غير متاح مع الفلتر الحالي.") + f" ({type(e).__name__})")
            i += 2
    disclaimer_box()


@st.cache_data(show_spinner=False)
def seg_models(_Xp, sig, k):
    pca_full = PCA(n_components=min(30, _Xp.shape[1]), random_state=42).fit(_Xp)
    Z = PCA(n_components=min(10, _Xp.shape[1]), random_state=42).fit_transform(_Xp)
    ks = list(range(2, 9))
    inertia, sil = [], []
    samp = np.random.RandomState(1).choice(len(Z), min(3000, len(Z)), replace=False)
    for kk in ks:
        km = KMeans(kk, n_init=5, random_state=42).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z[samp], km.labels_[samp]))
    lab = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
    bic = [GaussianMixture(kk, random_state=42).fit(Z).bic(Z) for kk in ks]
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(Z)
    return dict(evr=pca_full.explained_variance_ratio_, Z=Z, ks=ks, inertia=inertia, sil=sil, lab=lab, bic=bic,
                score=-iso.score_samples(Z), flag=iso.predict(Z) == -1)


@st.cache_data(show_spinner=False)
def tsne_emb(_Z, sig):
    idx = np.random.RandomState(0).choice(len(_Z), min(2500, len(_Z)), replace=False)
    return idx, TSNE(2, perplexity=35, random_state=42, init="pca").fit_transform(_Z[idx])


def page_seg(d, art):
    st.markdown(f'<div class="sec-t">{mi("bubble_chart")} {tr("Unsupervised learning – patient segmentation & anomalies", "التعلّم غير المُشرَف – تقسيم المرضى والشواذ")}</div>', unsafe_allow_html=True)
    k = st.slider(tr("Number of clusters (K-Means)", "عدد المجموعات (K-Means)"), 2, 8, 4, key="kclusters")
    sig = f"{len(d)}"
    Xp = processed_all(art, d, sig)
    with st.spinner(tr("Computing segments…", "جاري حساب المجموعات…")):
        R = seg_models(Xp, sig, k)
    t = d.assign(Cluster=[f"C{c + 1}" for c in R["lab"]], Anom=R["flag"], Score=R["score"])
    prof = t.groupby("Cluster").agg(n=("Cluster", "size"), ha=(TARGET, "mean"), age=("AgeMidpoint", "mean"), bmi=("BMI", "mean"),
                                     chronic=("ChronicCount", "mean"), diab=("HasDiabetes", "mean"), smk=("CurrentSmoker", "mean"),
                                     health=("HealthScore", "mean"))
    worst = prof["ha"].idxmax()
    kpis([("extension", tr("Clusters", "المجموعات"), str(k), tr("K-Means on PCA", "K-Means على PCA"), CYAN),
          ("straighten", tr("PCA comps for 90%", "مكونات PCA لـ90%"), str(int(np.searchsorted(np.cumsum(R['evr']), 0.9) + 1)), tr("variance explained", "من التباين"), VIO),
          ("emergency", tr("Highest-risk cluster", "أخطر مجموعة"), worst, pct(prof.loc[worst, "ha"] * 100), RED),
          ("shield", tr("Silhouette", "معامل Silhouette"), f"{R['sil'][R['ks'].index(k)]:.2f}", tr("cluster separation", "فصل المجموعات"), GRN),
          ("search", tr("Anomalies (5%)", "الشواذ (5%)"), f"{int(R['flag'].sum()):,}", "Isolation Forest", YEL),
          ("favorite", tr("Anomaly HA rate", "معدل النوبات في الشواذ"), pct(t.loc[t.Anom, TARGET].mean() * 100), tr("vs", "مقابل") + f" {pct(t[TARGET].mean() * 100)}", PINK)])
    cs = [GRN, CYAN, VIO, YEL, PINK, BLUE, RED, TEAL]
    a, b = st.columns(2)
    with a, st.container(key="card_seg_pca"):
        card_header(tr("PCA cumulative explained variance", "التباين التراكمي المُفسَّر لـ PCA"), "trending_up")
        cum = np.cumsum(R["evr"]) * 100
        f = go.Figure(go.Scatter(x=list(range(1, len(cum) + 1)), y=cum, mode="lines+markers", line=dict(color=CYAN, width=3), fill="tozeroy"))
        f.add_hline(y=90, line_dash="dash", line_color=YEL)
        f.update_layout(xaxis_title=tr("Components", "المكونات"), yaxis_title="%")
        show(f, "seg_pca")
    with b, st.container(key="card_seg_k"):
        card_header(tr("Choosing K: elbow & silhouette", "اختيار K: الكوع و Silhouette"), "straighten")
        f = go.Figure()
        f.add_trace(go.Scatter(x=R["ks"], y=R["inertia"], name=tr("Inertia", "Inertia"), line=dict(color=VIO, width=3)))
        f.add_trace(go.Scatter(x=R["ks"], y=R["sil"], name="Silhouette", yaxis="y2", line=dict(color=GRN, width=3)))
        f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), xaxis_title="K")
        show(f, "seg_k")
    a, b = st.columns(2)
    with a, st.container(key="card_seg_rate"):
        card_header(tr("Heart-attack rate per cluster (%)", "معدل النوبات القلبية لكل مجموعة (%)"), "favorite")
        s = prof["ha"] * 100
        f = px.bar(x=s.index, y=s.values, color=s.values, color_continuous_scale=RISK_SCALE, text=[f"{v:.1f}%" for v in s.values])
        f.update_layout(coloraxis_showscale=False)
        show(f, "seg_rate")
    with b, st.container(key="card_seg_gmm"):
        card_header(tr("GMM BIC curve", "منحنى BIC لـ GMM"), "grid_on")
        f = go.Figure(go.Scatter(x=R["ks"], y=R["bic"], mode="lines+markers", line=dict(color=PINK, width=3)))
        f.update_layout(xaxis_title="K", yaxis_title="BIC")
        show(f, "seg_gmm")
    with st.container(key="card_seg_prof"):
        card_header(tr("Cluster profile heatmap (z-score)", "خريطة خصائص المجموعات (z-score)"), "local_fire_department")
        cols = ["age", "bmi", "chronic", "diab", "smk", "health", "ha"]
        names = [tr("Age", "العمر"), "BMI", tr("Chronic", "مزمنة"), tr("Diabetes", "سكري"), tr("Smoking", "تدخين"), tr("Health", "الصحة"), tr("Heart attack", "نوبة")]
        z = (prof[cols] - prof[cols].mean()) / prof[cols].std(ddof=0).replace(0, 1)
        f = px.imshow(z.values, x=names, y=list(prof.index), color_continuous_scale="RdBu_r", zmin=-2, zmax=2, text_auto=".2f", aspect="auto")
        show(f, "seg_prof", 340)
    a, b = st.columns(2)
    with a, st.container(key="card_seg_scatter"):
        card_header(tr("Clusters in PCA space", "المجموعات في فضاء PCA"), "scatter_plot")
        idx = np.random.RandomState(3).choice(len(t), min(4000, len(t)), replace=False)
        f = px.scatter(x=R["Z"][idx, 0], y=R["Z"][idx, 1], color=t["Cluster"].values[idx], opacity=.65, color_discrete_sequence=cs,
                       labels={"x": "PC1", "y": "PC2", "color": ""})
        show(f, "seg_scatter", 420)
    with b, st.container(key="card_seg_iso"):
        card_header(tr("Isolation-Forest anomaly scores", "درجات الشواذ – Isolation Forest"), "emergency")
        f = px.histogram(t, x="Score", color=t[TARGET].map({0: tr("No history", "بدون"), 1: tr("Heart attack", "نوبة")}), nbins=50,
                         barmode="overlay", opacity=.7, color_discrete_sequence=[GRN, RED])
        show(f, "seg_iso", 420)
    with st.container(key="card_seg_tsne"):
        card_header(tr("t-SNE projection (2,500 patients sample)", "إسقاط t-SNE (عينة 2500 مريض)"), "blur_on")
        if st.button(tr("Compute t-SNE", "احسب t-SNE"), key="tsnebtn"):
            with st.spinner("t-SNE…"):
                idx, E = tsne_emb(R["Z"], sig)
            st.session_state["tsne"] = (idx, E)
        if "tsne" in st.session_state:
            idx, E = st.session_state["tsne"]
            f = px.scatter(x=E[:, 0], y=E[:, 1], color=t["Cluster"].values[idx], opacity=.7, color_discrete_sequence=cs, labels={"color": ""})
            show(f, "seg_tsne", 480)
    disclaimer_box()


def page_models(d, art):
    st.markdown(f'<div class="sec-t">{mi("smart_toy")} {tr("Supervised models – evaluation", "النماذج المُشرَفة – التقييم")}</div>', unsafe_allow_html=True)
    y, X = art["yte"], art["Xte"]
    p = art["model"].predict_proba(art["pre"].transform(X))[:, 1]
    roc, pr = roc_auc_score(y, p), average_precision_score(y, p)
    src = tr("notebook artifacts", "ملفات النوتبوك") if art["source"] == "notebook" else tr("re-trained in app (same pipeline)", "أُعيد تدريبه داخل التطبيق (نفس الـpipeline)")
    st.markdown(f'<span class="chip">{mi("psychology")} {art["name"]}</span><span class="chip">{mi("inventory_2")} {src}</span><span class="chip">{mi("science")} {tr("held-out test set", "مجموعة الاختبار")} n={len(y):,}</span>', unsafe_allow_html=True)
    thr = st.slider(tr("Decision threshold", "عتبة القرار"), 0.05, 0.95, float(round(art["thr"], 2)), 0.01, key="thr_slider",
                    help=tr("Lower threshold catches more cases (higher recall) but more false alarms.", "عتبة أقل تعني اكتشاف حالات أكثر (Recall أعلى) لكن إنذارات كاذبة أكثر."))
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    rec, prc = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    kpis([("trending_up", "ROC-AUC", f"{roc:.3f}", tr("ranking quality", "جودة الترتيب"), CYAN), ("target", "PR-AUC", f"{pr:.3f}", tr("imbalance-aware", "مناسب للبيانات غير المتوازنة"), VIO),
          ("shield", tr("Recall", "الحساسية Recall"), pct(rec * 100), f"{tp}/{tp + fn} {tr('caught', 'تم اكتشافها')}", GRN),
          ("biotech", tr("Precision", "الدقة Precision"), pct(prc * 100), f"{fp} {tr('false alarms', 'إنذار كاذب')}", YEL),
          ("flag", tr("Flagged", "نسبة المُنبَّه"), pct((tp + fp) / len(y) * 100), tr("of patients", "من المرضى"), PINK),
          ("tune", tr("Threshold", "العتبة"), f"{thr:.2f}", tr("notebook default", "الافتراضي") + f" {art['thr']:.2f}", RED)])
    a, b = st.columns(2)
    fpr, tpr, _ = roc_curve(y, p)
    with a, st.container(key="card_m_roc"):
        card_header("ROC", "trending_up")
        f = go.Figure(go.Scatter(x=fpr, y=tpr, fill="tozeroy", line=dict(color=CYAN, width=3), name="ROC"))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=mu_c()), name=tr("Chance", "عشوائي")))
        f.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        show(f, "m_roc")
    pp, rr, _ = precision_recall_curve(y, p)
    with b, st.container(key="card_m_pr"):
        card_header(tr("Precision–Recall", "Precision–Recall"), "target")
        f = go.Figure(go.Scatter(x=rr, y=pp, fill="tozeroy", line=dict(color=VIO, width=3)))
        f.add_hline(y=float(y.mean()), line_dash="dash", line_color=YEL, annotation_text=tr("baseline", "خط الأساس"))
        f.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        show(f, "m_pr")
    a, b = st.columns(2)
    with a, st.container(key="card_m_cm"):
        card_header(tr("Confusion matrix at the chosen threshold", "مصفوفة الالتباس عند العتبة المختارة"), "grid_on")
        z = np.array([[tn, fp], [fn, tp]])
        f = px.imshow(z, x=[tr("Pred: No", "توقع: لا"), tr("Pred: Yes", "توقع: نعم")], y=[tr("Actual: No", "فعلي: لا"), tr("Actual: Yes", "فعلي: نعم")],
                      text_auto=True, color_continuous_scale=[[0, "#16223b" if DARK else "#e2e8f0"], [1, CYAN]])
        f.update_layout(coloraxis_showscale=False)
        show(f, "m_cm")
    with b, st.container(key="card_m_trade"):
        card_header(tr("Threshold trade-off", "المفاضلة بين الحساسية والدقة"), "monitor_weight")
        ts = np.linspace(0.05, 0.95, 91)
        rcs = [((p >= t_) & (y == 1)).sum() / max((y == 1).sum(), 1) for t_ in ts]
        pcs = [((p >= t_) & (y == 1)).sum() / max((p >= t_).sum(), 1) for t_ in ts]
        f = go.Figure()
        f.add_trace(go.Scatter(x=ts, y=rcs, name="Recall", line=dict(color=GRN, width=3)))
        f.add_trace(go.Scatter(x=ts, y=pcs, name="Precision", line=dict(color=YEL, width=3)))
        f.add_vline(x=thr, line_dash="dash", line_color=RED)
        f.update_layout(xaxis_title=tr("Threshold", "العتبة"))
        show(f, "m_trade")
    a, b = st.columns(2)
    with a, st.container(key="card_m_dist"):
        card_header(tr("Score distribution by true class", "توزيع الدرجات حسب الفئة الحقيقية"), "bar_chart")
        f = px.histogram(x=p, color=np.where(y == 1, tr("Heart attack", "نوبة"), tr("No history", "بدون")), nbins=50, barmode="overlay",
                         opacity=.7, color_discrete_sequence=[GRN, RED], labels={"x": tr("Risk score", "درجة الخطر")})
        f.add_vline(x=thr, line_dash="dash", line_color=YEL)
        show(f, "m_dist")
    with b, st.container(key="card_m_cal"):
        card_header(tr("Calibration", "المعايرة Calibration"), "square_foot")
        q = pd.qcut(p, 8, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
        f = go.Figure(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", line=dict(color=PINK, width=3), marker=dict(size=9)))
        f.add_trace(go.Scatter(x=[0, g["p"].max()], y=[0, g["p"].max()], line=dict(dash="dash", color=mu_c()), name="ideal"))
        f.update_layout(xaxis_title=tr("Mean score", "متوسط الدرجة"), yaxis_title=tr("Observed rate", "المعدل الفعلي"), showlegend=False)
        show(f, "m_cal")
    a, b = st.columns(2)
    with a, st.container(key="card_m_board"):
        card_header(tr("Model comparison (5-fold CV PR-AUC, from the notebook)", "مقارنة النماذج (CV PR-AUC من النوتبوك)"), "emoji_events")
        board = pd.DataFrame({"Model": ["Logistic Regression", "LightGBM", "CatBoost", "XGBoost", "Random Forest"],
                              "CV PR-AUC": [0.4988, 0.4891, 0.4881, 0.4859, 0.4692]})
        f = px.bar(board, x="CV PR-AUC", y="Model", orientation="h", color="CV PR-AUC", color_continuous_scale="Tealgrn", text="CV PR-AUC")
        f.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"), xaxis_range=[0.4, 0.52])
        show(f, "m_board", 360)
    with b, st.container(key="card_m_coef"):
        card_header(tr("Top drivers of the score (model coefficients)", "أهم محركات الدرجة (معاملات النموذج)"), "explore")
        if hasattr(art["model"], "coef_"):
            names = art["pre"].get_feature_names_out()
            co = pd.Series(art["model"].coef_.ravel(), index=names)
            co = pd.concat([co.sort_values().head(7), co.sort_values().tail(7)])
            f = px.bar(x=co.values, y=co.index, orientation="h", color=co.values, color_continuous_scale=RISK_SCALE[::-1])
            f.update_layout(coloraxis_showscale=False, yaxis_title="")
            show(f, "m_coef", 360)
        else:
            st.info(tr("Coefficients are available for linear models only.", "المعاملات متاحة للنماذج الخطية فقط."))
    disclaimer_box()


def mu_c():
    return "#94a3b8"


def sel(label, key, options, index=0):
    return st.selectbox(label, options, index=index, key=key, format_func=disp)


def page_predict(d, art):
    st.markdown(f'<div class="sec-t">{mi("stethoscope")} {tr("Risk prediction demo", "تجربة توقع المخاطر")}</div>', unsafe_allow_html=True)
    disclaimer_box()
    slot = st.container()

    def opts(col):
        return [v for v in DEFAULTS[col] if v in set(d[col].unique())] or sorted(d[col].unique()) if col in d else DEFAULTS[col]

    with st.form("predict_form", border=False):
        c1, c2 = st.columns(2)
        with c1:
            with st.container(key="blk_demo"):
                st.markdown(f'<div class="blk-h"><span class="ic">{mi("person")}</span>{tr("Demographics", "البيانات الشخصية")}</div>', unsafe_allow_html=True)
                x1, x2 = st.columns(2)
                with x1:
                    sex = sel(tr("Sex", "الجنس"), "f_sex", opts("Sex"))
                with x2:
                    age = st.selectbox(tr("Age group", "الفئة العمرية"), AGE_ORDER, index=7, key="f_age", format_func=disp)
                race = sel(tr("Race / ethnicity", "العرق"), "f_race", opts("RaceEthnicityCategory"))
            with st.container(key="blk_body"):
                st.markdown(f'<div class="blk-h"><span class="ic">{mi("monitor_weight")}</span>{tr("Body measurements", "القياسات الجسمية")}</div>', unsafe_allow_html=True)
                x1, x2 = st.columns(2)
                h = x1.number_input(tr("Height (cm)", "الطول (سم)"), 120, 220, 170, key="f_h")
                w = x2.number_input(tr("Weight (kg)", "الوزن (كجم)"), 30, 250, 80, key="f_w")
                gh = st.select_slider(tr("General health (self-reported)", "الحالة الصحية العامة (تقييم ذاتي)"), GH_ORDER, value="Good", key="f_gh", format_func=disp)
            with st.container(key="blk_life"):
                st.markdown(f'<div class="blk-h"><span class="ic">{mi("smoking_rooms")}</span>{tr("Lifestyle", "نمط الحياة")}</div>', unsafe_allow_html=True)
                smoker = sel(tr("Smoking status", "حالة التدخين"), "f_smk", opts("SmokerStatus"))
                ecig = sel(tr("E-cigarette use", "السيجارة الإلكترونية"), "f_ecig", opts("ECigaretteUsage"))
                alc = st.toggle(tr("Drinks alcohol", "يشرب الكحول"), key="f_alc")
                hrl = st.toggle(tr("High-risk behaviour last year", "سلوك عالي الخطورة العام الماضي"), key="f_hrl")
        with c2:
            with st.container(key="blk_med"):
                st.markdown(f'<div class="blk-h"><span class="ic">{mi("stethoscope")}</span>{tr("Medical history", "التاريخ المرضي")}</div>', unsafe_allow_html=True)
                diab = sel(tr("Diabetes", "السكري"), "f_dia", opts("HadDiabetes"))
                t1, t2 = st.columns(2)
                ang = t1.toggle(tr("Angina", "الذبحة الصدرية"), key="f_ang")
                stk = t2.toggle(tr("Stroke", "سكتة دماغية"), key="f_stk")
                cop = t1.toggle("COPD", key="f_cop")
                ast = t2.toggle(tr("Asthma", "ربو"), key="f_ast")
                kid = t1.toggle(tr("Kidney disease", "أمراض الكلى"), key="f_kid")
                art_ = t2.toggle(tr("Arthritis", "التهاب المفاصل"), key="f_art")
                dep = t1.toggle(tr("Depressive disorder", "اضطراب اكتئابي"), key="f_dep")
                skc = t2.toggle(tr("Skin cancer", "سرطان جلد"), key="f_skc")
            with st.container(key="blk_func"):
                st.markdown(f'<div class="blk-h"><span class="ic">{mi("accessible")}</span>{tr("Daily function", "الأداء اليومي")}</div>', unsafe_allow_html=True)
                t1, t2 = st.columns(2)
                dh = t1.toggle(tr("Hearing difficulty", "صعوبة السمع"), key="f_dh")
                bl = t2.toggle(tr("Vision difficulty", "صعوبة الرؤية"), key="f_bl")
                dc = t1.toggle(tr("Difficulty concentrating", "صعوبة التركيز"), key="f_dc")
                dw = t2.toggle(tr("Difficulty walking", "صعوبة المشي"), key="f_dw")
                dd = t1.toggle(tr("Difficulty dressing/bathing", "صعوبة ارتداء الملابس/الاستحمام"), key="f_dd")
                de = t2.toggle(tr("Difficulty with errands", "صعوبة قضاء المشاوير"), key="f_de")
            with st.container(key="blk_prev"):
                st.markdown(f'<div class="blk-h"><span class="ic">{mi("vaccines")}</span>{tr("Prevention & tests", "الوقاية والفحوصات")}</div>', unsafe_allow_html=True)
                t1, t2 = st.columns(2)
                cs_ = t1.toggle(tr("Chest scan", "أشعة على الصدر"), key="f_cs")
                hiv = t2.toggle(tr("HIV test", "تحليل HIV"), key="f_hiv")
                flu = t1.toggle(tr("Flu vaccine (12m)", "تطعيم الإنفلونزا (12 شهراً)"), key="f_flu")
                pne = t2.toggle(tr("Pneumonia vaccine", "تطعيم الالتهاب الرئوي"), key="f_pne")
                cov = t1.toggle(tr("COVID-19 positive", "إصابة كوفيد-19"), key="f_cov")
                tet = sel(tr("Tetanus (last 10y)", "تطعيم التيتانوس (آخر 10 سنوات)"), "f_tet", opts("TetanusLast10Tdap"))
        go_ = st.form_submit_button(tr("Predict risk", "توقّع الخطورة"), width="stretch", icon=":material/ecg_heart:")

    if go_:
        bmi_v = w / ((h / 100) ** 2)
        row = pd.DataFrame([{
            "Sex": sex, "AgeCategory": age, "HeightInMeters": h / 100, "WeightInKilograms": w, "BMI": round(bmi_v, 1),
            "HadAngina": int(ang), "HadStroke": int(stk), "HadAsthma": int(ast), "HadSkinCancer": int(skc), "HadCOPD": int(cop),
            "HadDepressiveDisorder": int(dep), "HadKidneyDisease": int(kid), "HadArthritis": int(art_), "HadDiabetes": diab,
            "DeafOrHardOfHearing": int(dh), "BlindOrVisionDifficulty": int(bl), "DifficultyConcentrating": int(dc),
            "DifficultyWalking": int(dw), "DifficultyDressingBathing": int(dd), "DifficultyErrands": int(de),
            "SmokerStatus": smoker, "ECigaretteUsage": ecig, "ChestScan": int(cs_), "RaceEthnicityCategory": race,
            "AlcoholDrinkers": int(alc), "HIVTesting": int(hiv), "FluVaxLast12": int(flu), "PneumoVaxEver": int(pne),
            "TetanusLast10Tdap": tet, "HighRiskLastYear": int(hrl), "CovidPos": int(cov), "GeneralHealth": gh}])
        row = engineer(row)
        xr = np.asarray(art["pre"].transform(row), dtype=float)
        score = float(art["model"].predict_proba(xr)[0, 1])
        contrib = None
        if hasattr(art["model"], "coef_"):
            names = art["pre"].get_feature_names_out()
            mean_x = processed_all(art, d, str(len(d))).mean(axis=0)
            c = art["model"].coef_.ravel() * (xr[0] - mean_x)
            contrib = pd.Series(c, index=names)
        st.session_state["result"] = dict(score=score, bmi=bmi_v, bmic=row["BMI_Category"].iloc[0], contrib=contrib,
                                          comorb=int(row["ComorbidityCount"].iloc[0]), life=int(row["LifestyleRiskScore"].iloc[0]),
                                          inputs=dict(smoker=smoker, ecig=ecig, diab=diab, alc=bool(alc), hrl=bool(hrl), dw=bool(dw),
                                                      gh=gh, age=age, ang=bool(ang), stk=bool(stk)))

    r = st.session_state.get("result")
    if r:
        with slot:
            thr = art["thr"]
            high = r["score"] >= thr
            col = RED if high else GRN
            title = tr("High-risk indicators", "مؤشرات خطورة مرتفعة") if high else tr("Low-risk indicators", "مؤشرات خطورة منخفضة")
            msg = (tr("Your inputs resemble patients with a recorded heart-attack history. Please talk to a physician.",
                      "مدخلاتك تشبه حالات لديها تاريخ نوبة قلبية مسجّل. يُفضَّل استشارة طبيب.") if high else
                   tr("Your inputs are closer to patients without a recorded history. Keep up healthy habits.",
                      "مدخلاتك أقرب لحالات بدون تاريخ نوبة مسجّل. استمر على العادات الصحية."))
            left, right = st.columns([1.3, 1])
            with left:
                st.markdown(f'<div class="res" style="--rc:{col}"><h2>{mi("warning" if high else "check_circle")} {title}</h2><div class="big">{r["score"] * 100:.0f}%</div>'
                            f'<div>{tr("Model risk score", "درجة خطورة النموذج")} · {tr("decision threshold", "عتبة القرار")} {thr * 100:.0f}%</div>'
                            f'<p style="margin-top:10px;font-weight:600">{msg}</p>'
                            f'<span class="chip">BMI {r["bmi"]:.1f} · {disp(r["bmic"])}</span>'
                            f'<span class="chip">{tr("Comorbidities", "أمراض مصاحبة")}: {r["comorb"]}</span>'
                            f'<span class="chip">{tr("Lifestyle risk", "خطورة نمط الحياة")}: {r["life"]}/3</span></div>', unsafe_allow_html=True)
                disclaimer_box()
            with right, st.container(key="card_res_gauge"):
                g = go.Figure(go.Indicator(mode="gauge+number", value=r["score"] * 100, number=dict(suffix="%", font=dict(size=44)),
                                           gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=col, thickness=.28),
                                                      steps=[dict(range=[0, thr * 100], color="rgba(34,197,94,.30)"),
                                                             dict(range=[thr * 100, 100], color="rgba(239,68,68,.30)")],
                                                      threshold=dict(line=dict(color="#fff" if DARK else "#111", width=4), value=thr * 100))))
                show(g, "res_gauge", 300)
            if r["contrib"] is not None:
                with st.container(key="card_res_drv"):
                    card_header(tr("What pushed the score up / down (vs. the average patient)", "ما الذي رفع / خفض الدرجة (مقارنة بالمريض المتوسط)"), "explore")
                    s = r["contrib"]
                    s = s.reindex(s.abs().sort_values(ascending=False).head(10).index).sort_values()
                    f = px.bar(x=s.values, y=s.index, orientation="h", color=np.where(s.values > 0, tr("raises", "يرفع"), tr("lowers", "يخفض")),
                               color_discrete_map={tr("raises", "يرفع"): RED, tr("lowers", "يخفض"): GRN}, labels={"x": "log-odds", "y": "", "color": ""})
                    show(f, "res_drv", 380)



# ════════════════════════════════════════════════════════════════════
#  INSIGHTS + RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def ratio_table(_d, lang, sig):
    d = _d
    def T(en, ar):
        return ar if lang == "ar" else en
    defs = [
        ("stroke", T("Stroke history", "تاريخ سكتة"), d["HadStroke"] == 1),
        ("angina", T("Angina", "ذبحة صدرية"), d["HadAngina"] == 1),
        ("kidney", T("Kidney disease", "أمراض الكلى"), d["HadKidneyDisease"] == 1),
        ("copd", "COPD", d["HadCOPD"] == 1),
        ("diabetes", T("Diabetes", "سكري"), d["HasDiabetes"] == 1),
        ("age65", T("Age 65+", "عمر 65+"), d["AgeMidpoint"] >= 65),
        ("walk", T("Difficulty walking", "صعوبة المشي"), d["DifficultyWalking"] == 1),
        ("health", T("Poor / fair health", "صحة ضعيفة / مقبولة"), d["GeneralHealth"].isin(["Poor", "Fair"])),
        ("comorb3", T("3+ comorbidities", "3+ أمراض مصاحبة"), d["ComorbidityCount"] >= 3),
        ("arthritis", T("Arthritis", "التهاب مفاصل"), d["HadArthritis"] == 1),
        ("smoker", T("Current smoker", "مدخن حالياً"), d["CurrentSmoker"] == 1),
        ("depression", T("Depression", "اكتئاب"), d["HadDepressiveDisorder"] == 1),
        ("obese", T("Obesity (BMI ≥ 30)", "سمنة (BMI ≥ 30)"), d["BMI"] >= 30),
        ("male", T("Male", "ذكر"), d["Sex"] == "Male"),
    ]
    rows = []
    for key, name, m in defs:
        m = m.fillna(False)
        if m.sum() < 30 or (~m).sum() < 30:
            continue
        base = d.loc[~m, TARGET].mean()
        if base > 0:
            rows.append(dict(key=key, factor=name, ratio=d.loc[m, TARGET].mean() / base,
                             n=int(m.sum()), rate=d.loc[m, TARGET].mean() * 100, base=base * 100))
    return pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)


def ins_card(icon, title, text, color=None):
    sty = f' style="border-inline-start:4px solid {color}"' if color else ""
    return (f'<div class="ins"{sty}><div style="font-weight:800;margin-bottom:4px">'
            f'{mi(icon, None, color)} {title}</div><div style="color:var(--mu)">{text}</div></div>')


def cards_grid(items, per_row=3):
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, it in zip(cols, items[i:i + per_row]):
            col.markdown(ins_card(*it), unsafe_allow_html=True)


def page_insights(d, art):
    st.markdown(f'<div class="sec-t">{mi("lightbulb")} {tr("Insights", "الرؤى والاستنتاجات")}</div>', unsafe_allow_html=True)
    T = ratio_table(d, LANG, str(len(d)))
    if T.empty:
        st.info(tr("Not enough data to compute insights.", "البيانات غير كافية لحساب الاستنتاجات."))
        return
    R = T.set_index("key")
    ag = d.groupby("Age", observed=True)[TARGET].mean().mul(100)
    top = T.iloc[0]
    kpis([("priority_high", tr("Strongest association", "أقوى ارتباط"), str(top["factor"]), f"{top['ratio']:.1f}× {tr('heart-attack rate', 'معدل النوبات')}", RED),
          ("trending_up", tr("Highest-risk age group", "أعلى فئة عمرية خطورة"), str(ag.idxmax()), pct(ag.max()), VIO),
          ("favorite", tr("Overall rate", "المعدل العام"), pct(d[TARGET].mean() * 100), f"{len(d):,} {tr('patients', 'مريض')}", CYAN),
          ("analytics", tr("Factors analysed", "عوامل تم تحليلها"), str(len(T)), tr("vs. those without", "مقارنة بمن لا يملكونها"), GRN)])

    items = []
    young = d.loc[d["AgeMidpoint"] < 45, TARGET].mean()
    old_ = d.loc[d["AgeMidpoint"] >= 65, TARGET].mean()
    txt = tr(f"Rates climb steadily with age and peak at {ag.idxmax()} ({ag.max():.1f}%).", f"المعدل يرتفع مع العمر ويبلغ ذروته عند {ag.idxmax()} ({ag.max():.1f}%).")
    if young > 0:
        txt += " " + tr(f"People 65+ are about {old_ / young:.0f}× as likely as those under 45 to report one.", f"من هم فوق 65 أكثر احتمالاً بحوالي {old_ / young:.0f} ضعف من تحت 45.")
    items.append(("trending_up", tr("Age is the biggest divider", "العمر هو أكبر فارق"), txt, RED))
    items.append(("priority_high", tr("Strongest single association", "أقوى ارتباط منفرد"),
                  tr(f"{top['factor']}: {top['rate']:.1f}% have a heart-attack history vs {top['base']:.1f}% without it ({top['ratio']:.1f}×).",
                     f"{top['factor']}: {top['rate']:.1f}% لديهم تاريخ نوبة مقابل {top['base']:.1f}% بدونه ({top['ratio']:.1f}×)."), VIO))
    if "smoker" in R.index:
        sm = R.loc["smoker"]
        former = d.loc[d["SmokerStatus"].str.contains("Former"), TARGET].mean() * 100
        never = d.loc[d["SmokerStatus"].str.contains("Never"), TARGET].mean() * 100
        items.append(("smoking_rooms", tr("Smoking", "التدخين"),
                      tr(f"Current smokers: {sm['rate']:.1f}% vs {sm['base']:.1f}% for everyone else. Never-smokers sit at {never:.1f}%, former smokers at {former:.1f}% (partly an age effect).",
                         f"المدخنون حالياً: {sm['rate']:.1f}% مقابل {sm['base']:.1f}% لبقية العينة. من لم يدخنوا: {never:.1f}%، والمدخنون سابقاً: {former:.1f}% (جزء منها بسبب العمر)."), YEL))
    bc = d.groupby("BMI_Category")[TARGET].mean().mul(100)
    if "Obese" in bc.index and "Normal" in bc.index:
        items.append(("monitor_weight", tr("Weight", "الوزن"),
                      tr(f"Obese: {bc['Obese']:.1f}% vs normal weight: {bc['Normal']:.1f}%.", f"السمنة: {bc['Obese']:.1f}% مقابل الوزن الطبيعي: {bc['Normal']:.1f}%."), CYAN))
    cm = d.groupby(d["ComorbidityCount"].clip(upper=3))[TARGET].mean().mul(100)
    if 0 in cm.index and 3 in cm.index:
        items.append(("stethoscope", tr("Conditions stack up", "الأمراض تتراكم"),
                      tr(f"No other chronic condition: {cm[0]:.1f}%. Three or more: {cm[3]:.1f}%.", f"بدون أمراض مزمنة أخرى: {cm[0]:.1f}%. ثلاثة أو أكثر: {cm[3]:.1f}%."), PINK))
    sx = d.groupby("Sex")[TARGET].mean().mul(100)
    items.append(("wc", tr("Sex difference", "الفرق بين الجنسين"),
                  tr("Rates by sex: ", "المعدل حسب الجنس: ") + " · ".join(f"{disp(k)} {v:.1f}%" for k, v in sx.items()), TEAL))
    gs = d.groupby("State")[TARGET].agg(["mean", "size"])
    gs = gs[gs["size"] >= 30]
    if len(gs) >= 2:
        hi, lo = gs["mean"].idxmax(), gs["mean"].idxmin()
        items.append(("map", tr("Geography", "الجغرافيا"),
                      tr(f"Highest rate: {hi} ({gs.loc[hi, 'mean'] * 100:.1f}%), lowest: {lo} ({gs.loc[lo, 'mean'] * 100:.1f}%) among states with 30+ patients.",
                         f"الأعلى: {hi} ({gs.loc[hi, 'mean'] * 100:.1f}%)، الأقل: {lo} ({gs.loc[lo, 'mean'] * 100:.1f}%) بين الولايات التي بها 30+ مريض."), GRN))
    if hasattr(art["model"], "coef_"):
        co = pd.Series(art["model"].coef_.ravel(), index=art["pre"].get_feature_names_out()).sort_values(ascending=False)
        names = ", ".join(str(n) for n in co.head(3).index)
        items.append(("psychology", tr("What the model leans on", "على ماذا يعتمد النموذج"),
                      tr(f"The strongest score-raising features are: {names}.", f"أقوى الميزات رفعاً للدرجة: {names}."), BLUE))
    items.append(("info", tr("Read with care", "اقرأها بحذر"),
                  tr("These are associations in a sample dataset, not proof of cause. Heart-attack history is self-reported and age mixes into many of the gaps above.",
                     "هذه ارتباطات في بيانات عينة وليست إثباتاً للسبب. تاريخ النوبة مُبلَّغ ذاتياً، والعمر يدخل في كثير من الفروق."), YEL))
    cards_grid(items)

    with st.container(key="card_ins_ratio"):
        card_header(tr("Heart-attack rate with vs. without each factor (×)", "معدل النوبات مع وبدون كل عامل (×)"), "analytics")
        s = T.set_index("factor")["ratio"].sort_values()
        f = px.bar(x=s.values, y=s.index, orientation="h", color=s.values, color_continuous_scale=RISK_SCALE,
                   text=[f"{v:.1f}×" for v in s.values], labels={"x": "×", "y": ""})
        f.add_vline(x=1, line_dash="dash", line_color=mu_c())
        f.update_layout(coloraxis_showscale=False)
        f.update_traces(textposition="outside")
        show(f, "ins_ratio", 470)
    with st.container(key="card_ins_tbl"):
        card_header(tr("Details", "التفاصيل"), "table_chart")
        t = T[["factor", "n", "rate", "base", "ratio"]].copy()
        t.columns = [tr("Factor", "العامل"), tr("Patients", "المرضى"), tr("Rate with (%)", "المعدل مع (%)"),
                     tr("Rate without (%)", "المعدل بدون (%)"), tr("Ratio ×", "النسبة ×")]
        table(t.round(2), hide_index=True)
    disclaimer_box()


def page_reco(d, art):
    st.markdown(f'<div class="sec-t">{mi("checklist")} {tr("Recommendations", "التوصيات")}</div>', unsafe_allow_html=True)
    T = ratio_table(d, LANG, str(len(d)))
    R = T.set_index("key")["ratio"].to_dict() if not T.empty else {}

    r = st.session_state.get("result")
    if r:
        inp, tips = r["inputs"], []
        if "Current" in inp["smoker"]:
            tips.append(("smoking_rooms", tr("Get support to quit smoking", "اطلب دعماً للإقلاع عن التدخين"), tr("Ask a doctor or pharmacist about cessation programmes; quitting benefits the heart and lungs at any age.", "اسأل طبيباً أو صيدلانياً عن برامج الإقلاع؛ الإقلاع مفيد للقلب والرئتين في أي عمر."), YEL))
        if "Never" not in inp["ecig"] and "Not at all" not in inp["ecig"]:
            tips.append(("cloud", tr("Cut back on vaping", "قلّل استخدام السيجارة الإلكترونية"), tr("E-cigarettes are not risk-free for the heart and lungs.", "السيجارة الإلكترونية ليست خالية من الضرر على القلب والرئتين."), YEL))
        if r["bmic"] in ("Overweight", "Obese"):
            tips.append(("monitor_weight", tr("Work towards a healthier weight", "اتجه نحو وزن أصح"), tr("Small, steady changes in food and daily movement are easier to keep than crash plans. A dietitian can help.", "تغييرات صغيرة ومستمرة في الأكل والحركة أسهل في الاستمرار من الحميات القاسية. أخصائي تغذية يستطيع المساعدة."), CYAN))
        if inp["diab"] != "No":
            tips.append(("water_drop", tr("Keep blood sugar in check", "حافظ على سكر الدم"), tr("Follow your care plan and keep regular follow-up and lab visits.", "التزم بخطة العلاج وحافظ على المتابعة والتحاليل الدورية."), PINK))
        if inp["alc"]:
            tips.append(("local_bar", tr("Keep alcohol moderate", "اجعل الكحول في أضيق الحدود"), tr("Less is better for heart health; ask your doctor what is safe for you.", "الأقل أفضل لصحة القلب؛ اسأل طبيبك عما يناسبك."), VIO))
        if inp["dw"] or inp["gh"] in ("Poor", "Fair"):
            tips.append(("directions_walk", tr("Build gentle, regular activity", "ابنِ نشاطاً بدنياً خفيفاً ومنتظماً"), tr("Start slowly and discuss a safe plan with your doctor if movement is limited.", "ابدأ ببطء وناقش خطة آمنة مع طبيبك إذا كانت حركتك محدودة."), GRN))
        if r["comorb"] >= 2 or inp["ang"] or inp["stk"]:
            tips.append(("stethoscope", tr("Coordinate your conditions", "نسّق متابعة حالاتك المرضية"), tr("Keep one updated list of conditions and treatments and share it at every visit.", "احتفظ بقائمة واحدة محدّثة بالحالات والعلاج وشاركها في كل زيارة."), PINK))
        if AGE_ORDER.index(inp["age"]) >= 7:
            tips.append(("event_available", tr("Schedule regular check-ups", "حدد مواعيد فحص دورية"), tr("Blood pressure, cholesterol and blood sugar checks at the intervals your doctor advises.", "قياس الضغط والكوليسترول والسكر بالفترات التي ينصح بها طبيبك."), BLUE))
        high = r["score"] >= art["thr"]
        if high:
            tips.insert(0, ("medical_services", tr("Talk to a doctor soon", "تحدّث مع طبيب قريباً"), tr("Your inputs resemble higher-risk patients in this sample. This is not a diagnosis; a clinician can assess you properly.", "مدخلاتك تشبه مرضى أعلى خطورة في هذه العينة. هذا ليس تشخيصاً؛ والطبيب يستطيع تقييمك بشكل صحيح."), RED))
        if not tips:
            tips.append(("check_circle", tr("Keep up your habits", "استمر على عاداتك"), tr("Nothing in your inputs stands out. Keep moving, sleeping well and attending routine check-ups.", "لا شيء بارز في مدخلاتك. استمر في الحركة والنوم الجيد والفحوصات الدورية."), GRN))
        st.markdown(f'<div class="card-h">{mi("person")} {tr("For your demo result", "حسب نتيجتك التجريبية")}</div>', unsafe_allow_html=True)
        cards_grid(tips)
    else:
        st.markdown(ins_card("ecg_heart", tr("Personalised tips", "نصائح مخصصة"),
                             tr("Run the prediction on the Predict page and tailored tips will appear here.", "جرّب صفحة التوقع وستظهر هنا نصائح مخصصة لك."), CYAN), unsafe_allow_html=True)

    def note(key):
        return (tr(f" In this data: {R[key]:.1f}× the heart-attack rate.", f" في هذه البيانات: {R[key]:.1f}× معدل النوبات.") if key in R else "")

    st.markdown(f'<div class="sec-t" style="margin-top:14px">{mi("health_and_safety")} {tr("General recommendations", "توصيات عامة")}</div>', unsafe_allow_html=True)
    gen = [
        ("smoking_rooms", tr("Avoid tobacco and vaping", "ابتعد عن التدخين والفيب"), tr("Not smoking is one of the most effective ways to protect the heart and lungs.", "عدم التدخين من أكثر الطرق فاعلية لحماية القلب والرئتين.") + note("smoker"), YEL),
        ("monitor_weight", tr("Aim for a healthy weight", "اسعَ لوزن صحي"), tr("Sustainable food and activity habits beat short-term diets.", "عادات غذاء ونشاط مستدامة أفضل من الحميات المؤقتة.") + note("obese"), CYAN),
        ("water_drop", tr("Watch blood sugar", "راقب سكر الدم"), tr("Get screened regularly, and follow your care plan if you have diabetes.", "افحص بانتظام والتزم بخطة العلاج إن كنت مصاباً بالسكري.") + note("diabetes"), PINK),
        ("directions_walk", tr("Stay active", "ابقَ نشيطاً"), tr("Regular moderate activity, built up gradually. Ask your doctor if movement is limited.", "نشاط معتدل منتظم يزداد تدريجياً. اسأل طبيبك إن كانت حركتك محدودة.") + note("walk"), GRN),
        ("psychology", tr("Look after mental health", "اهتم بصحتك النفسية"), tr("Low mood and stress affect habits and heart health; professional support helps.", "المزاج المنخفض والتوتر يؤثران على العادات وصحة القلب؛ والدعم المتخصص مفيد.") + note("depression"), VIO),
        ("stethoscope", tr("Manage chronic conditions", "تابع أمراضك المزمنة"), tr("Keep a single list of conditions and treatments and share it at every visit.", "احتفظ بقائمة واحدة بالحالات والعلاج وشاركها في كل زيارة.") + note("comorb3"), BLUE),
        ("event_available", tr("Regular check-ups", "فحوصات دورية"), tr("Blood pressure, cholesterol and sugar checks at the intervals your doctor recommends.", "قياس الضغط والكوليسترول والسكر بالفترات التي يوصي بها طبيبك.") + note("age65"), TEAL),
        ("vaccines", tr("Stay up to date on vaccines", "حافظ على تطعيماتك"), tr("Flu and pneumonia vaccines are especially useful with heart or lung conditions.", "تطعيما الإنفلونزا والالتهاب الرئوي مفيدان خصوصاً مع أمراض القلب أو الرئة."), GRN),
        ("emergency", tr("Know the warning signs", "اعرف علامات الإنذار"), tr("Chest pain or pressure, shortness of breath, pain spreading to the arm or jaw, cold sweat or nausea: call your local emergency number immediately.", "ألم أو ضغط في الصدر، ضيق نفس، ألم ينتشر للذراع أو الفك، عرق بارد أو غثيان: اتصل بالإسعاف فوراً."), RED),
    ]
    cards_grid(gen)
    disclaimer_box()


# ── Health assistant (logic lives in health_assistant.py) ──
SUGGESTIONS = [
    ("favorite", "What are the warning signs of a heart attack?", "ما علامات الإنذار للنوبة القلبية؟"),
    ("monitor_weight", "How does BMI relate to heart disease?", "ما علاقة مؤشر كتلة الجسم بأمراض القلب؟"),
    ("smoking_rooms", "How does smoking affect the heart and lungs?", "كيف يؤثر التدخين على القلب والرئتين؟"),
    ("water_drop", "How can I lower my blood pressure naturally?", "كيف أخفض ضغط الدم بطرق طبيعية؟"),
]


def gemini_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        k = ""
    return (k or os.environ.get("GEMINI_API_KEY", "")).strip()


def chat_context(d, art):
    ctx = (f"\n\nAPP CONTEXT (optional, use only if relevant): the user is exploring an educational dashboard on a sample dataset of "
           f"{len(d):,} patients where {d[TARGET].mean() * 100:.1f}% have a recorded heart-attack history. "
           f"The demo model is {art['name']} (educational only, not clinically validated).")
    r = st.session_state.get("result")
    if r:
        ctx += (f" The user just ran the demo predictor: model risk score {r['score'] * 100:.0f}% "
                f"({'above' if r['score'] >= art['thr'] else 'below'} the decision threshold {art['thr'] * 100:.0f}%), "
                f"BMI {r['bmi']:.1f} ({r['bmic']}), comorbidity count {r['comorb']}. "
                f"This is a toy score, so explain it as educational and never as a diagnosis.")
    return ctx


def is_rtl_text(t):
    """True when the message is mostly Arabic letters (so it should start from the right)."""
    ar = sum(1 for c in t if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F")
    la = sum(1 for c in t if c.isascii() and c.isalpha())
    return ar > 0 and ar >= la


def page_chat(d, art):
    st.markdown(f'<div class="sec-t">{mi("smart_toy")} {tr("Health assistant", "المساعد الصحي")}</div>', unsafe_allow_html=True)
    st.session_state.setdefault("chat", [])
    main_c, side_c = st.columns([2.3, 1])

    with side_c:
        with st.container(key="card_chat_set"):
            card_header(tr("Settings", "الإعدادات"), "tune")
            st.toggle(tr("Use dashboard context", "استخدم سياق الداشبورد"), value=True, key="chat_ctx",
                      help=tr("Shares the dataset summary and your last demo prediction with the assistant.", "يشارك المساعد ملخص البيانات وآخر نتيجة توقع جربتها."))
            st.button(tr("Clear chat", "مسح المحادثة"), key="chat_clear", icon=":material/delete:", width="stretch",
                      on_click=lambda: st.session_state.update(chat=[]))
        with st.container(key="card_chat_sug"):
            card_header(tr("Quick questions", "أسئلة سريعة"), "lightbulb")
            for i, (ic_, en, ar) in enumerate(SUGGESTIONS):
                st.button(tr(en, ar), key=f"sug{i}", icon=f":material/{ic_}:", width="stretch",
                          on_click=lambda q=tr(en, ar): st.session_state.update(pending_q=q))
            if st.session_state.get("result"):
                st.button(tr("Explain my prediction result", "اشرح لي نتيجة التوقع"), key="sug_res", icon=":material/ecg_heart:", width="stretch",
                          on_click=lambda: st.session_state.update(pending_q=tr("Explain my demo prediction result in simple words and what healthy habits matter most.", "اشرح لي نتيجة التوقع التجريبية ببساطة وما أهم العادات الصحية المؤثرة.")))
        st.markdown(f'<div class="disc" style="--yl:#ef4444;background:rgba(239,68,68,.10)">{mi("emergency")} '
                    + tr("Severe chest pain, trouble breathing or stroke signs? Call your local emergency number now.",
                         "ألم شديد في الصدر أو صعوبة في التنفس أو علامات جلطة؟ اتصل بالإسعاف فوراً.") + "</div>", unsafe_allow_html=True)

    with main_c:
        key = gemini_key()
        q = st.chat_input(tr("Ask a general health question…", "اسأل سؤالاً صحياً عاماً…"))
        if not q:
            q = st.session_state.pop("pending_q", None)
        if q:
            st.session_state["chat"].append({"role": "user", "content": q})
            if not key:
                dev_log("GEMINI_API_KEY not found. Put it in .streamlit/secrets.toml or the GEMINI_API_KEY environment variable, then restart.")
                st.session_state["chat"].append({"role": "assistant", "content": tr(
                    "The assistant is not available right now.", "المساعد غير متاح حالياً."), "err": True})
            else:
                ctx = chat_context(d, art) if st.session_state.get("chat_ctx", True) else ""
                sp = build_system_prompt(ctx, concise=True)
                with st.spinner(tr("Thinking…", "جاري التفكير…")):
                    ans, model, err = ask_health_assistant(st.session_state["chat"], key, sp)
                if ans:
                    st.session_state["chat"].append({"role": "assistant", "content": ans, "model": model})
                else:
                    msgs = {"nolib": tr("The assistant is not available right now.", "المساعد غير متاح حالياً."),
                            "key": tr("The assistant is not available right now.", "المساعد غير متاح حالياً."),
                            "quota": tr("API quota / rate limit reached. Wait a bit and try again.", "تم الوصول لحد الاستخدام. انتظر قليلاً ثم أعد المحاولة."),
                            "server": tr("Gemini is temporarily unavailable. Try again shortly.", "خدمة Gemini غير متاحة مؤقتاً. حاول بعد قليل.")}
                    st.session_state["chat"].append({"role": "assistant", "content": msgs.get(err, tr("Something went wrong. Please try again.", "حدث خطأ. حاول مرة أخرى.")), "err": True})
        hist = st.session_state["chat"]
        if not hist:
            st.markdown(f'<div class="ins">{mi("smart_toy", 28)}<br><b>{tr("Hi! I am your health assistant.", "أهلاً! أنا مساعدك الصحي.")}</b><br>'
                        f'<span style="color:var(--mu)">{tr("Ask about heart health, diabetes, nutrition, sleep, lab tests and more.", "اسأل عن صحة القلب والسكري والتغذية والنوم والتحاليل وغيرها.")}</span></div>', unsafe_allow_html=True)
        groups = []
        for i, m in enumerate(hist):
            if m["role"] == "user" or not groups:
                groups.append([])
            groups[-1].append((i, m))
        for g in reversed(groups):  # newest exchange first
            for i, m in g:
                with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/smart_toy:"):
                    with st.container(key=f"msg_{i}_{'rtl' if is_rtl_text(m['content']) else 'ltr'}"):
                        (st.warning if m.get("err") else st.markdown)(m["content"])
                        if m.get("model"):
                            st.caption(f"Gemini · {m['model']}")
        disclaimer_box()


def page_landing():
    st.markdown(f'<div class="hero"><h1>{tr("Upload the dataset to begin", "ارفع ملف البيانات للبدء")}</h1><p>'
                f'{tr("Place MedicalDataset.xlsx next to app.py (or in a data/ folder), or upload it below.", "ضع ملف MedicalDataset.xlsx بجوار app.py (أو في مجلد data) أو ارفعه هنا.")}</p>{ecg_svg()}</div>', unsafe_allow_html=True)
    up = st.file_uploader("MedicalDataset.xlsx / .csv", type=["xlsx", "xls", "csv"])
    if up is not None:
        raw = pd.read_excel(up) if up.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(up)
        st.session_state["uploaded"] = raw
        st.rerun()


# ════════════════════════════════════════════════════════════════════
#  8. MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    topbar()
    path = find_file(["MedicalDataset.xlsx", "MedicalDataset.csv"])
    raw = st.session_state.get("uploaded")
    if raw is None and path:
        raw = read_data(path)
    if raw is None:
        page_landing()
        return
    d = prepare(raw)
    page = nav()
    with st.spinner(tr("Preparing model…", "جاري تجهيز النموذج…")):
        art = get_artifacts(d, f"{len(d)}")
    if page == "home":
        page_home(d, art)
    elif page == "eda":
        page_eda(d)
    elif page == "ins":
        page_insights(d, art)
    elif page == "rec":
        page_reco(d, art)
    elif page == "seg":
        page_seg(d, art)
    elif page == "mdl":
        page_models(d, art)
    elif page == "chat":
        page_chat(d, art)
    else:
        page_predict(d, art)


main()
