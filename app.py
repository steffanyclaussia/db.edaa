import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.anova import AnovaRM

# ==========================================
# 1. TEMA & KONFIGURASI (WARNA PREMIUM)
# ==========================================
PALETTE = {
    "green": "#76944C",
    "light_green": "#C8DAA6",
    "cream": "#FBF5DB",
    "yellow": "#FFD21F",
    "grey": "#C0B6AC",
    "dark_text": "#2B2B2B",
    "white": "#FFFFFF"
}

QUAL_ORDER = ["Premium", "Medium", "Pecah"]
QUAL_COLORS = {
    "Premium": PALETTE["green"],
    "Medium": PALETTE["yellow"],
    "Pecah": PALETTE["grey"],
}

MONTHS_ID = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ------------------------------------------
# 2. LOGIKA DATA (TETAP SAMA)
# ------------------------------------------
def _read_raw_csv(file_like):
    return pd.read_csv(file_like, header=None)

def parse_harga_beras(file_like):
    raw = _read_raw_csv(file_like)
    month_row_idx, jan_pos = None, None
    for i in range(raw.shape[0]):
        row = raw.iloc[i].astype(str)
        for j in range(raw.shape[1]):
            if row.iloc[j].strip() == "Januari":
                month_row_idx, jan_pos = i, j
                break
        if month_row_idx is not None: break

    if month_row_idx is None:
        raise ValueError("Format tidak dikenali.")

    month_map = {}
    for j in range(jan_pos, raw.shape[1]):
        val = str(raw.iloc[month_row_idx, j]).strip()
        if val in MONTHS_ID: month_map[j] = val
        elif val.lower() in ["tahunan", "tahun"] and "Desember" in month_map.values(): break

    tahun = None
    for r in range(max(0, month_row_idx-3), month_row_idx+1):
        for c in range(raw.shape[1]):
            cell = raw.iloc[r, c]
            if pd.isna(cell): continue
            s = str(cell).strip()
            if s.isdigit() and len(s) == 4:
                tahun = int(s)
                break
        if tahun is not None: break

    records = []
    for i in range(month_row_idx+1, raw.shape[0]):
        qual = raw.iloc[i, 0]
        if pd.isna(qual): continue
        qual = str(qual).strip().title()
        for col_idx, month in month_map.items():
            val = raw.iloc[i, col_idx]
            if pd.isna(val): continue
            s = str(val).strip().replace(",", "")
            try: harga = float(s)
            except: continue
            records.append({"Tahun": tahun, "Bulan": month, "Kualitas": qual, "Harga": harga})

    long_df = pd.DataFrame(records)
    long_df = long_df[long_df["Kualitas"].isin(QUAL_ORDER)].copy()
    long_df["Bulan"] = long_df["Bulan"].astype(MONTH_CAT)
    return long_df.sort_values(["Bulan", "Kualitas"]).reset_index(drop=True), {"tahun": tahun}

def make_wide(long_df):
    return long_df.pivot_table(index="Bulan", columns="Kualitas", values="Harga", aggfunc="mean").reindex(columns=QUAL_ORDER)

def repeated_measures_anova(long_df):
    wide = make_wide(long_df).dropna()
    long_complete = wide.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
    aov = AnovaRM(long_complete, depvar="Harga", subject="Bulan", within=["Kualitas"]).fit()
    table = aov.anova_table
    f_val, p_val = table["F Value"].iloc[0], table["Pr > F"].iloc[0]
    df1, df2 = table["Num DF"].iloc[0], table["Den DF"].iloc[0]
    eta2 = (f_val * df1) / (f_val * df1 + df2)
    return {"table": table, "F": f_val, "p": p_val, "eta": eta2, "wide": wide}

def paired_posthoc(wide):
    import itertools
    pairs = list(itertools.combinations(wide.columns, 2))
    rows = []
    for a, b in pairs:
        x, y = wide[a].dropna(), wide[b].dropna()
        t, p = stats.ttest_rel(x, y)
        rows.append({"Pasangan": f"{a} vs {b}", "t": t, "p_raw": p, "Diff": x.mean() - y.mean()})
    return pd.DataFrame(rows)

# ==========================================
# 3. STREAMLIT UI (WARNA & LAYOUT BARU)
# ==========================================
st.set_page_config(page_title="Analytics Harga Beras", layout="wide")

# Custom CSS untuk efek Card dan warna latar
st.markdown(f"""
    <style>
    .stApp {{ background-color: {PALETTE['cream']}; }}
    div.block-container {{ padding-top: 2rem; }}
    
    /* Card Styling */
    .main-card {{
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        margin-bottom: 20px;
    }}
    
    /* Metric Styling */
    [data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid {PALETTE['light_green']};
        border-radius: 12px;
        padding: 15px;
    }}
    </style>
""", unsafe_allow_html=True)

# Header Dashboard
with st.container():
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.title("🌾 Dashboard Monitoring Harga Beras")
        st.markdown(f"**Analisis Tren Kualitas & Signifikansi Variansi Harga**")
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/1601/1601730.png", width=80)

st.divider()

# Sidebar Data
with st.sidebar:
    st.header("📂 Kelola Data")
    up = st.file_uploader("Upload CSV (Format BPS)", type=["csv"])
    use_sample = st.checkbox("Gunakan Data Contoh", value=(up is None))
    st.divider()
    mode = st.radio("Metode Uji Statistik", ["Repeated-Measures ANOVA", "One-Way ANOVA (Basic)"])

# Load Data
try:
    if up is not None and not use_sample:
        long_df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
        source_label = up.name
    else:
        # Gunakan path file Anda atau dummy data untuk testing
        long_df, meta = parse_harga_beras("data/Data_HargaBeras.csv")
        source_label = "Contoh_BPS.csv"
    wide = make_wide(long_df)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# Row 1: Key Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sumber Data", source_label)
m2.metric("Tahun Analisis", meta["tahun"] or "—")
m3.metric("Bulan Tercover", f"{len(wide)} Bulan")
m4.metric("Kualitas Beras", f"{len(long_df['Kualitas'].unique())} Tipe")

# Row 2: Tabs
tab1, tab2, tab3 = st.tabs(["📈 Visualisasi & Tren", "🧪 Hasil Uji Statistik", "📋 Data Mentah"])

with tab1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Pergerakan Harga Bulanan")
    # Line Chart dengan Line Shape 'Spline' (Lebih Halus)
    fig_line = px.line(
        long_df, x="Bulan", y="Harga", color="Kualitas",
        color_discrete_map=QUAL_COLORS, markers=True,
        line_shape="spline", # INI MEMBUAT GARIS TIDAK KAKU
        template="plotly_white"
    )
    fig_line.update_layout(hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("**Sebaran Harga (Boxplot)**")
        fig_box = px.box(long_df, x="Kualitas", y="Harga", color="Kualitas", 
                         color_discrete_map=QUAL_COLORS, points="all")
        fig_box.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.write("**Rata-rata Harga per Kualitas**")
        mean_df = long_df.groupby("Kualitas")["Harga"].mean().reset_index()
        fig_bar = px.bar(mean_df, x="Kualitas", y="Harga", color="Kualitas", 
                         color_discrete_map=QUAL_COLORS, text_auto='.0f')
        fig_bar.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Keputusan Uji Statistik")
    
    if "Repeated" in mode:
        rm = repeated_measures_anova(long_df)
        
        # Highlight Keputusan
        if rm["p"] < 0.05:
            st.success(f"✅ **Signifikan!** Terdapat perbedaan harga yang nyata antar kualitas (p = {rm['p']:.4g})")
        else:
            st.warning(f"⚠️ **Tidak Signifikan.** Tidak ada perbedaan harga yang nyata (p = {rm['p']:.4g})")
        
        c_s1, c_s2, c_s3 = st.columns(3)
        c_s1.metric("F-Statistic", f"{rm['F']:.3f}")
        c_s2.metric("P-Value", f"{rm['p']:.4f}")
        c_s3.metric("Effect Size (η²)", f"{rm['eta']:.3f}")

        st.divider()
        st.write("**Analisis Berpasangan (Post-hoc):**")
        post = paired_posthoc(rm["wide"])
        
        # Styling Tabel Posthoc (P-Value Highlight)
        def color_p(val):
            color = '#d4edda' if val < 0.05 else '#f8d7da'
            return f'background-color: {color}'
        
        st.dataframe(post.style.applymap(color_p, subset=['p_raw']).format({"t":"{:.3f}", "p_raw":"{:.4f}", "Diff":"{:,.2f}"}), use_container_width=True)
    
    else:
        st.info("Mode One-Way ANOVA berjalan. Hasil akan ditampilkan di sini.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("Tabel Data Referensi")
    st.dataframe(wide.style.format("{:,.0f}").background_gradient(cmap="Greens", axis=0), use_container_width=True)
    
    st.download_button(
        label="📥 Download Data Olahan (CSV)",
        data=long_df.to_csv(index=False).encode('utf-8'),
        file_name='data_harga_beras_clean.csv',
        mime='text/csv',
    )
