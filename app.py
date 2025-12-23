import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. KONFIGURASI TEMA & WARNA
# ==========================================
CHART_PALETTE = {
    "moss_green": "#76944C", "light_sage": "#C8DAA6", "cream": "#FBF5DB",
    "honey_yellow": "#FFD21F", "warm_grey": "#C0B6AC", "dark_text": "#2F3632",
}
QUAL_COLORS = {"Premium": "#76944C", "Medium": "#FFD21F", "Pecah": "#C0B6AC"}
MONTHS_ID = ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ==========================================
# 2. FUNGSI HELPER STATISTIK
# ==========================================
def keputusan_normal(p, alpha=0.05): return "NORMAL" if p >= alpha else "TIDAK NORMAL"
def keputusan_homogen(p, alpha=0.05): return "HOMOGEN" if p >= alpha else "TIDAK HOMOGEN"

def parse_harga_beras(file_like):
    raw = pd.read_csv(file_like, header=None)
    idx_bulan = raw.apply(lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(), axis=1).idxmax()
    month_map = {j: str(raw.iloc[idx_bulan, j]).strip() for j in range(1, 13) if str(raw.iloc[idx_bulan, j]).strip() in MONTHS_ID}
    tahun = next((int(str(raw.iloc[r, c]).strip()) for r in range(max(0, idx_bulan-3), idx_bulan+1) 
                  for c in range(raw.shape[1]) if str(raw.iloc[r, c]).strip().isdigit() and len(str(raw.iloc[r, c]).strip()) == 4), 2024)
    records = []
    for i in range(idx_bulan+1, raw.shape[0]):
        qual = str(raw.iloc[i, 0]).strip().title()
        if qual not in ["Premium", "Medium", "Pecah"]: continue
        for col_idx, month in month_map.items():
            val = str(raw.iloc[i, col_idx]).strip().replace(",", "")
            try: records.append({"kualitas": qual, "bulan": month, "harga": float(val)})
            except: continue
    df = pd.DataFrame(records)
    df["bulan"] = df["bulan"].astype(MONTH_CAT)
    return df.sort_values(["bulan", "kualitas"]).reset_index(drop=True), tahun

# ==========================================
# 3. UI DASHBOARD
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras RCBD", layout="wide")

st.markdown(f"""
<style>
    html, body, [class*="st-"], .stMarkdown, .stTable, .stDataFrame {{ font-family: "Times New Roman", Times, serif !important; }}
    .stApp {{ background-color: {CHART_PALETTE['cream']}; color: {CHART_PALETTE['dark_text']}; }}
    .header-box {{ background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%); padding: 2.5rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem; }}
    .stats-card {{ background-color: white; padding: 20px; border-radius: 15px; border: 1px solid {CHART_PALETTE['warm_grey']}; margin-bottom: 20px; }}
    code {{ color: #d63384; }}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><h1>Dashboard Analisis Harga Beras</h1><p>Laporan Statistik Formal: Randomized Complete Block Design (RCBD)</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 Pengaturan Data")
    up = st.file_uploader("Unggah File CSV (BPS)", type=["csv"])

if up:
    long, tahun = parse_harga_beras(io.BytesIO(up.getvalue()))
    wide = long.pivot_table(index="bulan", columns="kualitas", values="harga")
    
    t1, t2, t3, t4 = st.tabs(["📋 Data Check", "📊 EDA Deskriptif", "🧪 Uji Asumsi", "🔬 Hasil ANOVA & Post-Hoc"])

    # --- TAB 1: DATA CHECK ---
    with t1:
        st.markdown("### === DATA CHECK ===")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.text(f"Bulan: {list(MONTHS_ID)}")
            st.text(f"Wide shape: {wide.T.shape} | Long shape: {long.shape}")
            st.text(f"Kualitas: {sorted(long['kualitas'].unique().tolist())}")
        with col_c2:
            st.text("Jumlah data per kualitas:")
            st.write(long.groupby("kualitas")["harga"].size().to_dict())
            st.text(f"Duplikat (bulan, kualitas): {long.duplicated(subset=['bulan', 'kualitas']).sum()}")
        
        st.markdown("**Preview long:**")
        st.dataframe(long.head(12), use_container_width=True)

    # --- TAB 2: EDA ---
    with t2:
        st.markdown("### === EDA (Deskriptif harga) ===")
        desc = long.groupby("kualitas")["harga"].agg(["count", "mean", "std", "min", "max"])
        st.table(desc.style.format("{:.2f}"))

        fig_line = px.line(long, x="bulan", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS, markers=True, title="Trend Harga Beras")
        st.plotly_chart(fig_line, use_container_width=True)

    # --- TAB 3: UJI ASUMSI ---
    with t3:
        st.markdown("### === NORMALITAS DATA per POPULASI (Shapiro) ===")
        norm_res = []
        for k, g in long.groupby("kualitas"):
            p = stats.shapiro(g["harga"].to_numpy()).pvalue
            norm_res.append({"Kualitas": k, "p-value": f"{p:.7f}", "Keputusan": keputusan_normal(p)})
        st.table(pd.DataFrame(norm_res))

        # Model for Residuals
        model = ols('harga ~ C(kualitas) + C(bulan)', data=long).fit()
        resid_stud = model.get_influence().resid_studentized_internal
        
        st.markdown("### === HOMOGENITAS VARIANSI (Residual) ===")
        groups = [g["harga"].to_numpy() for _, g in long.groupby("kualitas")] # Levene on raw groups for variance
        p_lev = stats.levene(*groups).pvalue
        st.write(f"**Levene p-value:** `{p_lev:.6f}` | **Status:** `{keputusan_homogen(p_lev)}`")

        st.markdown("### === NORMALITAS RESIDUAL MODEL (Shapiro) ===")
        p_sh = stats.shapiro(resid_stud).pvalue
        st.write(f"**Shapiro residual p-value:** `{p_sh:.6f}` | **Status:** `{keputusan_normal(p_sh)}`")

    # --- TAB 4: ANOVA & POST-HOC ---
    with t4:
        st.markdown("### === ANOVA (harga ~ kualitas + bulan) ===")
        aov_table = anova_lm(model, typ=2)
        st.table(aov_table.style.format("{:.6e}"))

        # Effect Size
        ss_res = aov_table.loc["Residual", "sum_sq"]
        eta_qual = aov_table.loc["C(kualitas)", "sum_sq"] / (aov_table.loc["C(kualitas)", "sum_sq"] + ss_res)
        eta_month = aov_table.loc["C(bulan)", "sum_sq"] / (aov_table.loc["C(bulan)", "sum_sq"] + ss_res)
        
        st.markdown("### === EFFECT SIZE (Partial Eta Squared) ===")
        st.write(f"Partial Eta^2 Kualitas: `{eta_qual:.4f}`")
        st.write(f"Partial Eta^2 Bulan: `{eta_month:.4f}`")

        st.markdown("### === POST-HOC KUALITAS (Holm) ===")
        # Pairwise T-Tests (Model Based)
        pairs = [("Pecah", "Medium", "C(kualitas)[T.Pecah]"), 
                 ("Premium", "Medium", "C(kualitas)[T.Premium]"),
                 ("Premium", "Pecah", "Premium_vs_Pecah")]
        
        ph_results = []
        p_values = []
        
        # Pecah vs Medium
        t_pec_med = model.t_test("C(kualitas)[T.Pecah] = 0")
        ph_results.append(["Pecah - Medium", t_pec_med.effect[0][0], t_pec_med.tvalue[0][0], t_pec_med.pvalue])
        p_values.append(t_pec_med.pvalue)

        # Premium vs Medium
        t_pre_med = model.t_test("C(kualitas)[T.Premium] = 0")
        ph_results.append(["Premium - Medium", t_pre_med.effect[0][0], t_pre_med.tvalue[0][0], t_pre_med.pvalue])
        p_values.append(t_pre_med.pvalue)

        # Premium vs Pecah
        t_pre_pec = model.t_test("C(kualitas)[T.Premium] - C(kualitas)[T.Pecah] = 0")
        ph_results.append(["Premium - Pecah", t_pre_pec.effect[0][0], t_pre_pec.tvalue[0][0], t_pre_pec.pvalue])
        p_values.append(t_pre_pec.pvalue)

        # Adjust P-values
        rej, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="holm")
        
        final_ph = []
        for i, res in enumerate(ph_results):
            final_ph.append({
                "Pasangan": res[0], "diff": f"{res[1]:.4f}", "t": f"{res[2]:.3f}", 
                "p": f"{res[3]:.6e}", "p_adj": f"{p_adj[i]:.6e}", "signif": rej[i]
            })
        st.table(pd.DataFrame(final_ph))

else:
    st.info("👋 Selamat Datang. Silakan unggah file CSV data harga beras Anda.")
