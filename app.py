import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy import stats
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
QUAL_COLORS = {"Premium": "#76944C", "Medium": "#FFD21F", "Pecah": "#C0B6AC"}
MONTHS_ID = ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ==========================================
# 2. FUNGSI PEMROSESAN DATA (DIPERBAIKI)
# ==========================================
def parse_harga_beras(file_like):
    raw = pd.read_csv(file_like, header=None)
    
    # Mencari baris yang mengandung 'Januari'
    idx_bulan = raw.apply(lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(), axis=1).idxmax()
    
    # Mapping kolom ke bulan
    month_map = {}
    for j in range(1, raw.shape[1]):
        val = str(raw.iloc[idx_bulan, j]).strip()
        if val in MONTHS_ID:
            month_map[j] = val

    # Mencari Tahun
    tahun = 2024
    tahun_search = raw.iloc[max(0, idx_bulan-3):idx_bulan+1, :].astype(str)
    for val in tahun_search.values.flatten():
        if val.isdigit() and len(val) == 4:
            tahun = int(val)
            break

    records = []
    for i in range(idx_bulan+1, raw.shape[0]):
        qual = str(raw.iloc[i, 0]).strip().title()
        if qual not in QUAL_ORDER: continue
        for col_idx, month in month_map.items():
            val = str(raw.iloc[i, col_idx]).strip().replace(",", "")
            try:
                records.append({"kualitas": qual, "bulan": month, "harga": float(val)})
            except: continue

    df = pd.DataFrame(records)
    df["bulan"] = df["bulan"].astype(MONTH_CAT)
    return df.sort_values(["bulan", "kualitas"]).reset_index(drop=True), tahun

# ==========================================
# 3. STREAMLIT UI (TIMES NEW ROMAN)
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
    }}
    /* Perbaikan agar Tahun 2024 tidak terpotong */
    div[data-testid="stMetric"] {{
        background: white; border-radius: 15px; padding: 20px; 
        border-bottom: 6px solid {CHART_PALETTE['honey_yellow']};
        min-width: 250px !important;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.8rem !important; white-space: nowrap !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="header-box"><h1 style="margin:0;">Dashboard Analisis Harga Beras</h1></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Pengaturan")
    up = st.file_uploader("Upload CSV BPS", type=["csv"])

if up:
    df_long, tahun = parse_harga_beras(io.BytesIO(up.getvalue()))
    df_wide = df_long.pivot_table(index="bulan", columns="kualitas", values="harga").reindex(columns=QUAL_ORDER)

    # Metrics Section
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Periode Data", f"Tahun {tahun}")
    m2.metric("Rerata Premium", f"Rp {df_long[df_long['kualitas']=='Premium']['harga'].mean():,.0f}")
    m3.metric("Rerata Medium", f"Rp {df_long[df_long['kualitas']=='Medium']['harga'].mean():,.0f}")
    m4.metric("Rerata Pecah", f"Rp {df_long[df_long['kualitas']=='Pecah']['harga'].mean():,.0f}")

    # Tabs Section
    t1, t2, t3 = st.tabs(["📈 Tren & Distribusi", "📋 Matriks Data", "🔬 Statistik RCBD"])

    with t1:
        st.plotly_chart(px.line(df_long, x="bulan", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS, markers=True, title="Tren Harga"), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.box(df_long, x="kualitas", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS), use_container_width=True)
        with c2: st.plotly_chart(px.bar(df_long.groupby("kualitas")['harga'].mean().reset_index(), x="kualitas", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS), use_container_width=True)

    with t2:
        st.write("### Matriks Harga Bulanan")
        st.dataframe(df_wide.style.format("{:,.0f}"), use_container_width=True)

    with t3:
        st.write("### Analisis Inferensial (ANOVA RCBD)")
        model = ols('harga ~ C(kualitas) + C(bulan)', data=df_long).fit()
        aov_table = anova_lm(model, typ=2)
        aov_table.index = ['C(kualitas)', 'C(bulan)', 'Residual']
        st.table(aov_table.style.format({"sum_sq": "{:.6e}", "df": "{:.1f}", "F": "{:.6f}", "PR(>F)": "{:.6e}"}))

        # Post-Hoc Holm
        st.write("### Post-Hoc Test (Holm)")
        hypotheses = [("Pecah - Medium", "C(kualitas)[T.Pecah] = 0"), 
                      ("Premium - Medium", "C(kualitas)[T.Premium] = 0"),
                      ("Premium - Pecah", "C(kualitas)[T.Premium] - C(kualitas)[T.Pecah] = 0")]
        
        ph_results = []
        raw_p = []
        for label, hyp in hypotheses:
            t_test = model.t_test(hyp)
            ph_results.append({"Pasangan": label, "diff": t_test.effect.item(), "t": t_test.tvalue.item(), "p": t_test.pvalue.item()})
            raw_p.append(t_test.pvalue.item())
        
        rej, p_adj, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        for i, res in enumerate(ph_results):
            res["p_adj"] = p_adj[i]
            res["signif"] = rej[i]
        
        st.table(pd.DataFrame(ph_results).style.format({"diff": "{:.2f}", "t": "{:.3f}", "p": "{:.4e}", "p_adj": "{:.4e}"}))

else:
    st.info("👋 Selamat Datang. Silakan unggah file CSV Anda melalui panel di samping kiri untuk melihat analisis.")
