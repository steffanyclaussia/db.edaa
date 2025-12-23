import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Import library untuk ANOVA RCBD (OLS Model)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. KONFIGURASI TEMA & WARNA (PALETTE)
# ==========================================
CHART_PALETTE = {
    "moss_green": "#76944C",    
    "light_sage": "#C8DAA6",    
    "cream": "#FBF5DB",         
    "honey_yellow": "#FFD21F",  
    "warm_grey": "#C0B6AC",     
    "dark_text": "#2F3632",     
}

QUAL_ORDER = ["Premium", "Medium", "Pecah"]
QUAL_COLORS = {
    "Premium": "#76944C",      
    "Medium": "#FFD21F",       
    "Pecah": "#C0B6AC",        
}

MONTHS_ID = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ==========================================
# 2. FUNGSI PEMROSESAN DATA
# ==========================================
def parse_harga_beras(file_like) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(file_like, header=None)
    month_row_idx = None
    jan_pos = None
    for i in range(raw.shape[0]):
        row = raw.iloc[i].astype(str)
        for j in range(raw.shape[1]):
            if row.iloc[j].strip() == "Januari":
                month_row_idx, jan_pos = i, j
                break
        if month_row_idx is not None: break

    if month_row_idx is None:
        raise ValueError("Format BPS tidak ditemukan.")

    month_map = {j: str(raw.iloc[month_row_idx, j]).strip() for j in range(jan_pos, raw.shape[1]) 
                 if str(raw.iloc[month_row_idx, j]).strip() in MONTHS_ID}
    
    tahun = next((int(str(raw.iloc[r, c]).strip()) for r in range(max(0, month_row_idx-3), month_row_idx+1) 
                  for c in range(raw.shape[1]) if str(raw.iloc[r, c]).strip().isdigit() and len(str(raw.iloc[r, c]).strip()) == 4), 2024)

    records = []
    for i in range(month_row_idx+1, raw.shape[0]):
        qual = str(raw.iloc[i, 0]).strip()
        if not qual or qual.lower() == "nan": continue
        for col_idx, month in month_map.items():
            val = str(raw.iloc[i, col_idx]).strip().replace(",", "")
            try:
                records.append({"Tahun": tahun, "Bulan": month, "Kualitas": qual.title(), "Harga": float(val)})
            except: continue

    df = pd.DataFrame(records)
    df = df[df["Kualitas"].isin(QUAL_ORDER)].copy()
    df["Bulan"] = df["Bulan"].astype("string").astype(MONTH_CAT)
    return df.sort_values(["Bulan", "Kualitas"]).reset_index(drop=True), {"tahun": tahun}

def make_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot_table(index="Bulan", columns="Kualitas", values="Harga", aggfunc="mean").reindex(columns=QUAL_ORDER)

# ==========================================
# 3. STREAMLIT UI 
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras RCBD", layout="wide")

st.markdown(f"""
<style>
    html, body, [class*="st-"], .stMarkdown, .stTable, .stDataFrame {{
        font-family: "Times New Roman", Times, serif !important;
    }}
    .stApp {{ background-color: {CHART_PALETTE['cream']}; color: {CHART_PALETTE['dark_text']}; }}
    .header-box {{
        background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%);
        padding: 2.5rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }}
    div[data-testid="stMetric"] {{
        background: white; border-radius: 15px; padding: 20px; 
        border-bottom: 6px solid {CHART_PALETTE['honey_yellow']};
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .stats-card {{
        background-color: white; padding: 25px; border-radius: 15px;
        border: 1px solid {CHART_PALETTE['warm_grey']}; margin-bottom: 20px;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""<div class="header-box">
    <h1 style="margin:0;">Dashboard Analisis Harga Beras</h1>
    <p style="font-size: 1.2rem; font-style: italic;">Laporan Statistik Formal: RCBD</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 Pengaturan Data")
    up = st.file_uploader("Unggah File CSV", type=["csv"])

if up:
    df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
    wide_df = make_wide(df)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Periode", f"Tahun {meta['tahun']}")
    with m2: st.metric("Rerata Premium", f"Rp {df[df['Kualitas']=='Premium']['Harga'].mean():,.0f}")
    with m3: st.metric("Rerata Medium", f"Rp {df[df['Kualitas']=='Medium']['Harga'].mean():,.0f}")
    with m4: st.metric("Rerata Pecah", f"Rp {df[df['Kualitas']=='Pecah']['Harga'].mean():,.0f}")

    t1, t2, t3 = st.tabs(["📈 Tren & Distribusi", "📋 Matriks Data", "🔬 Statistik Mendalam"])

    with t3:
        st.markdown("<h2 style='text-align: center;'>Analisis Inferensial (RCBD)</h2>", unsafe_allow_html=True)
        
        # Penjelasan model
        long_complete = wide_df.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
        model = ols('Harga ~ C(Kualitas) + C(Bulan)', data=long_complete).fit()
        aov_table = anova_lm(model, typ=2)
        
        # PERBAIKAN FORMAT TABEL ANOVA
        aov_display = aov_table.copy()
        aov_display.index = ['C(kualitas)', 'C(bulan)', 'Residual']
        
        st.markdown("#### === ANOVA (harga ~ kualitas + bulan) ===")
        st.table(aov_display.style.format({
            "sum_sq": "{:.6e}", "df": "{:.1f}", "F": "{:.6f}", "PR(>F)": "{:.6e}"
        }))

        # PERBAIKAN POST-HOC (HOLM)
        st.markdown("#### === POST-HOC KUALITAS (Holm) ===")
        
        hypotheses = [
            ("Pecah - Medium", "C(Kualitas)[T.Pecah] = 0"),
            ("Premium - Medium", "C(Kualitas)[T.Premium] = 0"),
            ("Premium - Pecah", "C(Kualitas)[T.Premium] - C(Kualitas)[T.Pecah] = 0")
        ]
        
        ph_rows = []
        raw_pvals = []
        for label, hyp in hypotheses:
            t_test = model.t_test(hyp)
            # PERBAIKAN: Menggunakan .item() untuk mengekstrak nilai skalar dari array NumPy
            diff_val = t_test.effect.item() if hasattr(t_test.effect, 'item') else float(t_test.effect)
            t_val = t_test.tvalue.item() if hasattr(t_test.tvalue, 'item') else float(t_test.tvalue)
            p_val = t_test.pvalue.item() if hasattr(t_test.pvalue, 'item') else float(t_test.pvalue)
            
            ph_rows.append({
                "Pasangan": label,
                "diff": diff_val,
                "t": t_val,
                "p": p_val
            })
            raw_pvals.append(p_val)
            
        rej, p_adj, _, _ = multipletests(raw_pvals, alpha=0.05, method="holm")
        
        for i, row in enumerate(ph_rows):
            row["p_adj"] = p_adj[i]
            row["signif"] = rej[i]
            
        st.table(pd.DataFrame(ph_rows).style.format({
            "diff": "{:.4f}", "t": "{:.3f}", "p": "{:.6e}", "p_adj": "{:.6e}"
        }))
        
else:
    st.info("👋 Silakan unggah file CSV data harga beras.")
