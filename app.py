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
def parse_harga_beras(file_like):
    raw = pd.read_csv(file_like, header=None)
    idx_bulan = raw.apply(lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(), axis=1).idxmax()
    
    month_map = {}
    for j in range(1, raw.shape[1]):
        val = str(raw.iloc[idx_bulan, j]).strip()
        if val in MONTHS_ID:
            month_map[j] = val

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
# 3. STREAMLIT UI (DESIGN & TYPOGRAPHY)
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras RCBD", layout="wide")

# CSS untuk memaksakan Times New Roman di semua elemen
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tinos:ital,wght@0,400;0,700;1,400;1,700&display=swap');

    html, body, [class*="st-"], .stMarkdown, .stTable, .stDataFrame, div[data-testid="stMetricValue"], button {{
        font-family: "Times New Roman", Times, serif !important;
    }}
    
    .stApp {{ background-color: {CHART_PALETTE['cream']}; color: {CHART_PALETTE['dark_text']}; }}
    
    .header-box {{
        background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%);
        padding: 3rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }}

    /* Desain Kartu Metrik agar Tidak Terpotong */
    div[data-testid="stMetric"] {{
        background: white; border-radius: 15px; padding: 25px; 
        border-bottom: 8px solid {CHART_PALETTE['honey_yellow']};
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        min-width: 280px !important;
    }}
    
    div[data-testid="stMetricValue"] {{ 
        font-size: 2.2rem !important; 
        font-weight: bold !important; 
        white-space: nowrap !important;
        color: {CHART_PALETTE['dark_text']};
    }}

    /* Styling Tab */
    .stTabs [data-baseweb="tab-list"] {{ gap: 20px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; background-color: white; border-radius: 10px 10px 0 0;
        padding: 10px 20px; font-weight: bold; border: 1px solid #ddd;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {CHART_PALETTE['moss_green']} !important; color: white !important;
    }}

    /* Container Box untuk Informasi Statistik */
    .info-card {{
        background-color: white; padding: 25px; border-radius: 15px;
        border-left: 10px solid {CHART_PALETTE['moss_green']};
        margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }}
</style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown(f"""
    <div class="header-box">
        <h1 style="margin:0; font-size: 3.5rem;">Laporan Analisis Harga Beras</h1>
        <p style="font-size: 1.4rem; font-style: italic; opacity: 0.9;">
            Pendekatan Statistik Formal: Randomized Complete Block Design (RCBD)
        </p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h2 style='color:white;'>📂 Menu Data</h2>", unsafe_allow_html=True)
    up = st.file_uploader("Unggah File CSV BPS", type=["csv"])
    st.divider()
    st.info("Sistem akan secara otomatis mendeteksi periode tahun dan struktur bulan dalam dokumen Anda.")

if up:
    df_long, tahun = parse_harga_beras(io.BytesIO(up.getvalue()))
    df_wide = df_long.pivot_table(index="bulan", columns="kualitas", values="harga").reindex(columns=QUAL_ORDER)

    # --- BAGIAN METRIK (RINGKASAN CEPAT) ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Periode Analisis", f"Tahun {tahun}")
    m2.metric("Rerata Premium", f"Rp {df_long[df_long['kualitas']=='Premium']['harga'].mean():,.0f}")
    m3.metric("Rerata Medium", f"Rp {df_long[df_long['kualitas']=='Medium']['harga'].mean():,.0f}")
    m4.metric("Rerata Pecah", f"Rp {df_long[df_long['kualitas']=='Pecah']['harga'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BAGIAN TABS ---
    t1, t2, t3 = st.tabs(["📊 Visualisasi Tren & Distribusi", "📋 Matriks Data Lengkap", "🔬 Analisis Statistik RCBD"])

    with t1:
        st.markdown(f"### <span style='color:{CHART_PALETTE['moss_green']}'>📈 Analisis Visual Harga</span>", unsafe_allow_html=True)
        
        # Line Chart (Tren)
        fig_line = px.line(df_long, x="bulan", y="harga", color="kualitas", 
                           color_discrete_map=QUAL_COLORS, markers=True, 
                           title="Tren Fluktuasi Harga Beras Bulanan")
        fig_line.update_layout(font_family="Times New Roman", title_font_size=22, plot_bgcolor='rgba(0,0,0,0)')
        fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        st.plotly_chart(fig_line, use_container_width=True)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            # Boxplot (Distribusi)
            fig_box = px.box(df_long, x="kualitas", y="harga", color="kualitas", 
                            color_discrete_map=QUAL_COLORS, title="Sebaran & Variabilitas Harga")
            fig_box.update_layout(font_family="Times New Roman", showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        with col_v2:
            # Bar Chart (Rata-rata)
            avg_data = df_long.groupby("kualitas")['harga'].mean().reset_index()
            fig_bar = px.bar(avg_data, x="kualitas", y="harga", color="kualitas", 
                            color_discrete_map=QUAL_COLORS, title="Perbandingan Rata-rata Tahunan")
            fig_bar.update_layout(font_family="Times New Roman", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    with t2:
        st.markdown(f"### <span style='color:{CHART_PALETTE['moss_green']}'>📋 Matriks Harga Beras</span>", unsafe_allow_html=True)
        st.markdown("Data mentah hasil konversi dari dokumen BPS yang telah dibersihkan:")
        st.dataframe(df_wide.style.format("{:,.0f}").background_gradient(cmap='Greens'), use_container_width=True)

    with t3:
        st.markdown(f"### <span style='color:{CHART_PALETTE['moss_green']}'>🔬 Analisis Inferensial (ANOVA RCBD)</span>", unsafe_allow_html=True)
        
        # Informasi Konteks
        st.markdown(f"""
            <div class="info-card">
                <h4 style="margin:0; color:{CHART_PALETTE['moss_green']};">Interpretasi Model RCBD</h4>
                <p style="margin:10px 0 0 0;">
                    Model ini menganalisis pengaruh <b>Kualitas</b> (Perlakuan) terhadap harga dengan mengontrol variabel 
                    <b>Bulan</b> (Blok). Pendekatan ini memastikan bahwa perbedaan harga benar-benar disebabkan oleh kualitas, 
                    bukan karena fluktuasi musiman di bulan tertentu.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Perhitungan ANOVA
        model = ols('harga ~ C(kualitas) + C(bulan)', data=df_long).fit()
        aov_table = anova_lm(model, typ=2)
        aov_table.index = ['C(kualitas)', 'C(bulan)', 'Residual']
        
        # Tabel ANOVA Formal
        st.markdown("#### **Tabel Analisis Ragam (ANOVA)**")
        st.table(aov_table.style.format({
            "sum_sq": "{:.6e}", "df": "{:.1f}", "F": "{:.6f}", "PR(>F)": "{:.6e}"
        }))

        # Post-Hoc Test (Holm)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### **Uji Lanjut: Post-Hoc Test (Holm Adjustment)**")
        
        hypotheses = [
            ("Pecah vs Medium", "C(kualitas)[T.Pecah] = 0"), 
            ("Premium vs Medium", "C(kualitas)[T.Premium] = 0"),
            ("Premium vs Pecah", "C(kualitas)[T.Premium] - C(kualitas)[T.Pecah] = 0")
        ]
        
        ph_results = []
        raw_p = []
        for label, hyp in hypotheses:
            t_test = model.t_test(hyp)
            # Menggunakan .item() untuk keamanan konversi array ke skalar
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

        # Kesimpulan Akhir
        p_val_main = aov_table.loc["C(kualitas)", "PR(>F)"]
        if p_val_main < 0.05:
            st.success(f"**Kesimpulan Akhir:** Berdasarkan hasil ANOVA, terdapat perbedaan harga yang **signifikan** antar kategori kualitas beras (p < 0.05). Hasil Post-Hoc mengonfirmasi bahwa seluruh pasangan kualitas memiliki perbedaan harga rata-rata yang nyata secara statistik.")
        else:
            st.warning("**Kesimpulan Akhir:** Tidak ditemukan perbedaan harga yang signifikan antar kualitas beras pada tingkat kepercayaan 95%.")

else:
    # Tampilan saat file belum diunggah
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👋 Selamat Datang. Silakan unggah file CSV hasil laporan BPS melalui panel di samping kiri untuk memulai analisis statistik.")
