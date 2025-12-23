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
# 2. FUNGSI PEMROSESAN DATA
# ==========================================
def parse_harga_beras(source):
    # Mendukung input berupa path string atau file-like object
    raw = pd.read_csv(source, header=None)
    idx_bulan = raw.apply(lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(), axis=1).idxmax()
    
    month_map = {j: str(raw.iloc[idx_bulan, j]).strip() for j in range(1, raw.shape[1]) 
                 if str(raw.iloc[idx_bulan, j]).strip() in MONTHS_ID}

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
# 3. STREAMLIT UI (FULL TIMES NEW ROMAN)
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras RCBD", layout="wide")

st.markdown(f"""
<style>
    /* Mengatur Font Times New Roman Secara Global */
    html, body, [class*="st-"], .stMarkdown, .stTable, .stDataFrame, 
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
    button, select, input, .stTabs, label, p, span, h1, h2, h3, h4 {{
        font-family: "Times New Roman", Times, serif !important;
    }}
    
    .stApp {{ background-color: {CHART_PALETTE['cream']}; color: {CHART_PALETTE['dark_text']}; }}
    
    .header-box {{
        background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%);
        padding: 2.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }}

    div[data-testid="stMetric"] {{
        background: white; border-radius: 12px; padding: 20px; 
        border-bottom: 6px solid {CHART_PALETTE['honey_yellow']};
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        min-width: 280px !important;
    }}

    .stTable, div[data-testid="stTable"], .stDataFrame {{
        background-color: #FFFFFF !important;
        border: 1px solid #D1D1D1 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {CHART_PALETTE['moss_green']} !important; color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="header-box">
        <h1 style="margin:0; font-size: 3.2rem; color: white;">Laporan Analisis Harga Beras</h1>
        <p style="font-size: 1.3rem; font-style: italic; opacity: 0.9; color: white;">Sistem Otomasi Data - Randomized Complete Block Design (RCBD)</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOGIKA AUTO-LOAD DATA
# ==========================================
data_source = None
DEFAULT_FILE = "Data_HargaBeras.csv" # Nama file yang akan dicari otomatis

with st.sidebar:
    st.markdown("### 📂 Pengaturan Data")
    # Coba cek apakah file default ada
    try:
        with open(DEFAULT_FILE, "rb") as f:
            data_source = DEFAULT_FILE
            st.success(f"✅ Otomatis menggunakan: {DEFAULT_FILE}")
    except FileNotFoundError:
        st.warning(f"⚠️ {DEFAULT_FILE} tidak ditemukan.")
    
    up = st.file_uploader("Unggah File CSV Baru (Opsional)", type=["csv"])
    if up is not None:
        data_source = io.BytesIO(up.getvalue())
        st.info("🔄 Menggunakan file yang baru diunggah.")

if data_source:
    df_long, tahun = parse_harga_beras(data_source)
    df_wide = df_long.pivot_table(index="bulan", columns="kualitas", values="harga").reindex(columns=QUAL_ORDER)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Periode Analisis", f"Tahun {tahun}")
    m2.metric("Rerata Premium", f"Rp {df_long[df_long['kualitas']=='Premium']['harga'].mean():,.0f}")
    m3.metric("Rerata Medium", f"Rp {df_long[df_long['kualitas']=='Medium']['harga'].mean():,.0f}")
    m4.metric("Rerata Pecah", f"Rp {df_long[df_long['kualitas']=='Pecah']['harga'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📈 Tren & Distribusi", "📋 Matriks Data", "🔬 Statistik RCBD"])

    with t1:
        st.markdown(f"### <span style='color:{CHART_PALETTE['moss_green']}'>Visualisasi Tren Harga</span>", unsafe_allow_html=True)
        fig_line = px.line(df_long, x="bulan", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS, markers=True)
        fig_line.update_layout(font_family="Times New Roman", font_color="#1A1A1A", plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(df_long, x="kualitas", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS)
            fig_box.update_layout(font_family="Times New Roman")
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            avg_df = df_long.groupby("kualitas")['harga'].mean().reset_index()
            fig_bar = px.bar(avg_df, x="kualitas", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS)
            fig_bar.update_layout(font_family="Times New Roman")
            st.plotly_chart(fig_bar, use_container_width=True)

    with t2:
        st.markdown("### 📋 Matriks Harga Beras")
        st.dataframe(df_wide.style.format("{:,.0f}").set_properties(**{'background-color': 'white', 'color': 'black'}), use_container_width=True)

    with t3:
        st.markdown(f"### <span style='color:{CHART_PALETTE['moss_green']}'>Analisis Inferensial (ANOVA RCBD)</span>", unsafe_allow_html=True)
        model = ols('harga ~ C(kualitas) + C(bulan)', data=df_long).fit()
        aov_table = anova_lm(model, typ=2)
        aov_table.index = ['C(kualitas)', 'C(bulan)', 'Residual']
        
        st.markdown("#### **Tabel Analisis Ragam (ANOVA)**")
        
        

        st.table(aov_table.style.format({
            "sum_sq": "{:.6e}", "df": "{:.1f}", "F": "{:.6f}", "PR(>F)": "{:.6e}"
        }))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### **Uji Lanjut: Post-Hoc Test (Holm Adjustment)**")
        
        hypotheses = [
            ("Pecah vs Medium", "C(kualitas)[T.Pecah] = 0"), 
            ("Premium vs Medium", "C(kualitas)[T.Premium] = 0"),
            ("Premium vs Pecah", "C(kualitas)[T.Premium] - C(kualitas)[T.Pecah] = 0")
        ]
        
        ph_results, raw_p = [], []
        for label, hyp in hypotheses:
            t_test = model.t_test(hyp)
            ph_results.append({
                "Perbandingan": label, 
                "Selisih": t_test.effect.item(), 
                "t-Stat": t_test.tvalue.item(), 
                "p-Value": t_test.pvalue.item()
            })
            raw_p.append(t_test.pvalue.item())
        
        rej, p_adj, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        for i, res in enumerate(ph_results):
            res["p-Adj (Holm)"] = p_adj[i]
            res["Signifikan"] = "Ya" if rej[i] else "Tidak"
        
        st.table(pd.DataFrame(ph_results).style.format({
            "Selisih": "{:.2f}", "t-Stat": "{:.3f}", "p-Value": "{:.4e}", "p-Adj (Holm)": "{:.4e}"
        }))

        if aov_table.loc["C(kualitas)", "PR(>F)"] < 0.05:
            st.success("**Kesimpulan:** Terdapat perbedaan harga signifikan antar kualitas beras (p < 0.05).")
        else:
            st.warning("**Kesimpulan:** Tidak ditemukan perbedaan harga signifikan antar kualitas beras.")

else:
    st.info("👋 Selamat Datang. Silakan letakkan file 'Data_HargaBeras.csv' di folder aplikasi atau unggah file secara manual.")
