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
                  for c in range(raw.shape[1]) if str(raw.iloc[r, c]).strip().isdigit() and len(str(raw.iloc[r, c]).strip()) == 4), None)

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
    /* Global Font Times New Roman */
    html, body, [class*="st-"], .stMarkdown, .stTable, .stDataFrame {{
        font-family: "Times New Roman", Times, serif !important;
    }}

    .stApp {{ 
        background-color: {CHART_PALETTE['cream']}; 
        color: {CHART_PALETTE['dark_text']}; 
    }}

    /* Header & Sidebar */
    .header-box {{
        background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%);
        padding: 2.5rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }}
    
    /* Metric Card Fix */
    div[data-testid="stMetric"] {{
        background: white; border-radius: 15px; padding: 20px; 
        border-bottom: 6px solid {CHART_PALETTE['honey_yellow']};
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        min-width: 200px !important;
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem !important; white-space: nowrap !important;
    }}

    /* Gaya Khusus Tabel Statistik */
    .stats-card {{
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid {CHART_PALETTE['warm_grey']};
        margin-bottom: 20px;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="header-box">
        <h1 style="font-family: 'Times New Roman', serif; font-size: 3rem; margin:0;">Dashboard Analisis Harga Beras</h1>
        <p style="font-size: 1.3rem; font-style: italic; margin-top:10px;">Laporan Statistik Formal: Randomized Complete Block Design (RCBD)</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 Pengaturan Data")
    up = st.file_uploader("Unggah File CSV (BPS)", type=["csv"])
    st.divider()
    st.caption("Font: Times New Roman | Palette: Moss Green & Honey Yellow")

if up:
    df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
    wide_df = make_wide(df)

    # Metrik Ringkasan
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric(label="Periode Analisis", value=f"Tahun {meta['tahun'] if meta['tahun'] else '2024'}")
    with m2: st.metric(label="Rata-rata Premium", value=f"Rp {df[df['Kualitas']=='Premium']['Harga'].mean():,.0f}")
    with m3: st.metric(label="Rata-rata Medium", value=f"Rp {df[df['Kualitas']=='Medium']['Harga'].mean():,.0f}")
    with m4: st.metric(label="Rata-rata Pecah", value=f"Rp {df[df['Kualitas']=='Pecah']['Harga'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📈 Tren & Distribusi", "📋 Matriks Data", "🔬 Statistik Mendalam"])

    with t1:
        fig_line = px.line(df, x="Bulan", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS, markers=True, title="Visualisasi Tren Harga Bulanan")
        fig_line.update_layout(font_family="Times New Roman", title_font_size=20)
        st.plotly_chart(fig_line, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(df, x="Kualitas", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS, title="Analisis Sebaran Harga")
            fig_box.update_layout(font_family="Times New Roman")
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            avg_df = df.groupby("Kualitas")["Harga"].mean().reset_index()
            fig_bar = px.bar(avg_df, x="Kualitas", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS, title="Perbandingan Harga Rata-rata")
            fig_bar.update_layout(font_family="Times New Roman")
            st.plotly_chart(fig_bar, use_container_width=True)

    with t2:
        st.markdown("### 📊 Matriks Harga Bulanan")
        st.dataframe(wide_df.style.highlight_max(axis=1, color=CHART_PALETTE['light_sage']).format("{:,.0f}"), use_container_width=True)

    with t3:
        st.markdown("<h2 style='text-align: center;'>Analisis Inferensial (RCBD)</h2>", unsafe_allow_html=True)
        
        # Penjelasan Konteks
        st.markdown(f"""
        <div class='stats-card'>
            <h4 style='color:{CHART_PALETTE['moss_green']};'>Konteks Model</h4>
            <p>Penelitian ini menggunakan <b>Rancangan Acak Kelompok (RCBD)</b> untuk menguji apakah terdapat perbedaan harga yang signifikan antar kualitas beras (Premium, Medium, Pecah) dengan memperlakukan <b>Bulan</b> sebagai blok untuk mengontrol variasi musiman.</p>
        </div>
        """, unsafe_allow_html=True)

        # Hitung ANOVA
        long_complete = wide_df.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
        model = ols('Harga ~ C(Kualitas) + C(Bulan)', data=long_complete).fit()
        aov_table = anova_lm(model, typ=2)
        aov_display = aov_table.rename(index={'C(Kualitas)': 'Kualitas (Perlakuan)', 'C(Bulan)': 'Bulan (Blok)', 'Residual': 'Error'})

        # Tampilan Tabel ANOVA yang Menarik
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("#### Tabel Analisis Ragam (ANOVA)")
            st.table(aov_display.style.format("{:.4f}").set_properties(**{
                'background-color': 'white',
                'color': CHART_PALETTE['dark_text'],
                'border-color': CHART_PALETTE['warm_grey']
            }))

        with col_right:
            st.markdown("#### Parameter Uji")
            p_val = aov_table.loc["C(Kualitas)", "PR(>F)"]
            f_stat = aov_table.loc["C(Kualitas)", "F"]
            
            st.markdown(f"""
            <div style='background-color:{CHART_PALETTE['moss_green']}; color:white; padding:20px; border-radius:10px; text-align:center;'>
                <small>P-Value (Kualitas)</small><br>
                <b style='font-size:24px;'>{p_val:.5f}</b><br><br>
                <small>F-Statistic</small><br>
                <b style='font-size:24px;'>{f_stat:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

        # Interpretasi Visual
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Interpretasi Hipotesis")
        
        if p_val < 0.05:
            st.markdown(f"""
            <div style='border-left: 10px solid {CHART_PALETTE['moss_green']}; background-color: #e8f5e9; padding: 20px;'>
                <h4 style='color:#2e7d32; margin:0;'>H<sub>1</sub> Diterima (Signifikan)</h4>
                <p style='margin:10px 0 0 0;'>Terdapat bukti statistik yang sangat kuat bahwa <b>setidaknya satu kategori kualitas memiliki harga rata-rata yang berbeda secara nyata</b> dibandingkan kategori lainnya pada tingkat kepercayaan 95%.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='border-left: 10px solid #c62828; background-color: #ffeae8; padding: 20px;'>
                <h4 style='color:#c62828; margin:0;'>H<sub>0</sub> Gagal Ditolak (Tidak Signifikan)</h4>
                <p style='margin:10px 0 0 0;'>Tidak ditemukan perbedaan harga yang signifikan antar kualitas beras. Variasi harga yang diamati kemungkinan besar hanya disebabkan oleh fluktuasi acak.</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👋 Selamat Datang. Silakan unggah data CSV Anda melalui panel di samping kiri.")
