import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ==========================================
# 1. KONFIGURASI TEMA & WARNA
# ==========================================
CHART_PALETTE = {
    "moss_green": "#76944C", "light_sage": "#C8DAA6", "cream": "#FBF5DB",
    "honey_yellow": "#FFD21F", "warm_grey": "#C0B6AC", "dark_text": "#2F3632",
}
QUAL_COLORS = {"Premium": "#76944C", "Medium": "#FFD21F", "Pecah": "#C0B6AC"}
MONTHS_ID = ["Januari","Februari","Maret","April","Mei","Juni",
             "Juli","Agustus","September","Oktober","November","Desember"]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ==========================================
# 2. UI & FONT TIMES NEW ROMAN
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras", layout="wide")

st.markdown(f"""
<style>
    html, body, [class*="st-"] {{ font-family: "Times New Roman", Times, serif !important; }}
    .stApp {{ background-color: {CHART_PALETTE['cream']}; color: {CHART_PALETTE['dark_text']}; }}
    .header-box {{
        background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%);
        padding: 2.5rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;
    }}
    div[data-testid="stMetric"] {{
        background: white; border-radius: 15px; padding: 20px; 
        border-bottom: 6px solid {CHART_PALETTE['honey_yellow']};
        min-width: fit-content !important;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.8rem !important; white-space: nowrap !important; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOGIKA PEMROSESAN DATA
# ==========================================
def process_data(file):
    raw = pd.read_csv(file, header=None)
    idx_bulan = raw.apply(lambda r: r.astype(str).str.contains(r"\bJanuari\b", na=False).any(), axis=1).idxmax()
    bulan_cols = raw.loc[idx_bulan, 1:12].tolist()
    
    wide = raw.loc[idx_bulan + 1:, [0] + list(range(1, 13))].copy()
    wide.columns = ["Kualitas"] + bulan_cols
    wide["Kualitas"] = wide["Kualitas"].astype(str).str.strip().str.title()
    wide = wide[wide["Kualitas"].isin(["Premium", "Medium", "Pecah"])].copy()
    
    for b in bulan_cols: wide[b] = pd.to_numeric(wide[b], errors="coerce")
    
    long = wide.melt(id_vars="Kualitas", var_name="Bulan", value_name="Harga").dropna()
    long["Bulan"] = pd.Categorical(long["Bulan"], categories=MONTHS_ID, ordered=True)
    return long.sort_values(["Bulan", "Kualitas"]), wide

# ==========================================
# 4. MAIN APP
# ==========================================
st.markdown('<div class="header-box"><h1>Dashboard Analisis Harga Beras</h1><p>Metode Statistik: Randomized Complete Block Design (RCBD)</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Menu Data")
    up = st.file_uploader("Upload CSV BPS", type=["csv"])

if up:
    df_long, df_wide = process_data(up)
    
    # Metrik Utama
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Periode Data", "Tahun 2024")
    m2.metric("Rerata Premium", f"Rp {df_long[df_long['Kualitas']=='Premium']['Harga'].mean():,.0f}")
    m3.metric("Rerata Medium", f"Rp {df_long[df_long['Kualitas']=='Medium']['Harga'].mean():,.0f}")
    m4.metric("Rerata Pecah", f"Rp {df_long[df_long['Kualitas']=='Pecah']['Harga'].mean():,.0f}")

    tab1, tab2, tab3 = st.tabs(["📊 Visualisasi", "📋 Matriks Data", "🔬 Statistik RCBD"])

    with tab1:
        fig_line = px.line(df_long, x="Bulan", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS, markers=True)
        fig_line.update_layout(font_family="Times New Roman", plot_bgcolor='white')
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.dataframe(df_wide.set_index("Kualitas").style.format("{:,.0f}"))

    with tab3:
        st.subheader("Analisis Ragam (ANOVA RCBD)")
        model = ols('Harga ~ C(Kualitas) + C(Bulan)', data=df_long).fit()
        aov_table = anova_lm(model, typ=2)
        
        st.table(aov_table.style.format("{:.4f}"))
        
        # Plot Residual menggunakan Matplotlib (Font Times New Roman)
        st.subheader("Uji Normalitas Residual")
        fig_res, ax_res = plt.subplots()
        stats.probplot(model.resid, dist="norm", plot=ax_res)
        ax_res.set_title("Normal Q-Q Plot", fontfamily="serif")
        plt.setp(ax_res.get_xticklabels(), fontfamily="serif")
        plt.setp(ax_res.get_yticklabels(), fontfamily="serif")
        st.pyplot(fig_res)

else:
    st.info("Silakan unggah file CSV data harga beras melalui sidebar.")
