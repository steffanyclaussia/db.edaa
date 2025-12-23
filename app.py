import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

# ==========================================
# 1. KONFIGURASI TEMA (FONT & COLOR)
# ==========================================
CHART_PALETTE = {
    "moss_green": "#76944C", "light_sage": "#C8DAA6", "cream": "#FBF5DB",
    "honey_yellow": "#FFD21F", "warm_grey": "#C0B6AC", "dark_text": "#2F3632"
}
QUAL_COLORS = {"Premium": "#76944C", "Medium": "#FFD21F", "Pecah": "#C0B6AC"}

# Font Times New Roman untuk Streamlit
st.markdown("""
<style>
    html, body, [class*="st-"] { font-family: "Times New Roman", Times, serif !important; }
    .stApp { background-color: #FBF5DB; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER & DATA PREPARATION
# ==========================================
def keputusan_normal(p, alpha=0.05): return "NORMAL" if p >= alpha else "TIDAK NORMAL"
def keputusan_homogen(p, alpha=0.05): return "HOMOGEN" if p >= alpha else "TIDAK HOMOGEN"

st.title("🔬 Analisis Statistik Harga Beras (Plotly Edition)")

# File Uploader (Ganti PATH lokal ke Uploader)
uploaded_file = st.sidebar.file_uploader("Upload Data_HargaBeras.csv", type=["csv"])

if uploaded_file:
    raw = pd.read_csv(uploaded_file, header=None)
    
    # Deteksi Baris Bulan
    idx_bulan = raw.apply(lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(), axis=1).idxmax()
    bulan = raw.loc[idx_bulan, 1:12].tolist()
    
    # Transformasi Wide ke Long
    wide = raw.loc[idx_bulan + 1:, [0] + list(range(1, 13))].copy()
    wide.columns = ["kualitas"] + bulan
    wide["kualitas"] = wide["kualitas"].astype(str).str.strip().str.title()
    wide = wide[wide["kualitas"].isin(["Premium", "Medium", "Pecah"])].copy()
    
    for b in bulan: wide[b] = pd.to_numeric(wide[b], errors="coerce")
    
    long = wide.melt(id_vars="kualitas", value_vars=bulan, var_name="bulan", value_name="harga").dropna()
    long["bulan"] = pd.Categorical(long["bulan"], categories=bulan, ordered=True)
    long = long.sort_values(["bulan", "kualitas"]).reset_index(drop=True)

    # ==========================================
    # 3. EDA DENGAN PLOTLY EXPRESS
    # ==========================================
    st.header("1. Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trend Plot (Line)
        fig_line = px.line(long, x="bulan", y="harga", color="kualitas", 
                           color_discrete_map=QUAL_COLORS, markers=True,
                           title="Trend Harga Beras (Jan–Des)")
        fig_line.update_layout(font_family="Times New Roman", plot_bgcolor='white')
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        # Box Plot
        fig_box = px.box(long, x="kualitas", y="harga", color="kualitas",
                         color_discrete_map=QUAL_COLORS,
                         title="Distribusi Harga per Kualitas")
        fig_box.update_layout(font_family="Times New Roman", plot_bgcolor='white')
        st.plotly_chart(fig_box, use_container_width=True)

    # ==========================================
    # 4. ASUMSI & ANOVA
    # ==========================================
    st.header("2. Analisis Statistik")
    
    model = smf.ols("harga ~ C(kualitas) + C(bulan)", data=long).fit()
    resid_stud = model.get_influence().resid_studentized_internal
    
    # Tabel ANOVA
    anova_tbl = anova_lm(model, typ=2)
    st.subheader("Tabel ANOVA (RCBD)")
    st.table(anova_tbl.style.format("{:.4f}"))

    # ==========================================
    # 5. DIAGNOSTIK DENGAN PLOTLY
    # ==========================================
    st.header("3. Diagnostik Model")
    
    c3, c4 = st.columns(2)
    
    with c3:
        # Q-Q Plot menggunakan Plotly
        qq_x, qq_y = stats.probplot(resid_stud, dist="norm")[0]
        fig_qq = px.scatter(x=qq_x, y=qq_y, title="Q-Q Plot Residual",
                            labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
        fig_qq.add_shape(type="line", x0=qq_x.min(), y0=qq_x.min(), x1=qq_x.max(), y1=qq_x.max(),
                         line=dict(color="Red", dash="dash"))
        fig_qq.update_layout(font_family="Times New Roman", plot_bgcolor='white')
        st.plotly_chart(fig_qq, use_container_width=True)

    with c4:
        # Residual vs Fitted
        fig_res = px.scatter(x=model.fittedvalues, y=resid_stud,
                             title="Residual vs Fitted",
                             labels={'x': 'Fitted Values', 'y': 'Studentized Residuals'})
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(font_family="Times New Roman", plot_bgcolor='white')
        st.plotly_chart(fig_res, use_container_width=True)

    # ==========================================
    # 6. POST-HOC TEST
    # ==========================================
    st.header("4. Post-Hoc Test (Holm)")
    
    # Logika post-hoc tetap sama (Holm Adjustment)
    names = model.params.index.tolist()
    def ttest_custom(weights):
        w = np.zeros(len(names))
        for nm, val in weights.items(): w[names.index(nm)] = val
        r = model.t_test(w)
        return float(r.effect), float(r.tvalue), float(r.pvalue)

    tests = [
        ("Pecah - Medium", {"C(kualitas)[T.Pecah]": 1.0}),
        ("Premium - Medium", {"C(kualitas)[T.Premium]": 1.0}),
        ("Premium - Pecah", {"C(kualitas)[T.Premium]": 1.0, "C(kualitas)[T.Pecah]": -1.0}),
    ]
    
    rows, pvals = [], []
    for label, wd in tests:
        eff, t, p = ttest_custom(wd)
        rows.append([label, eff, t, p])
        pvals.append(p)

    rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    
    results_df = pd.DataFrame(rows, columns=["Perbandingan", "Diff", "t-Stat", "p-Raw"])
    results_df["p-Adj (Holm)"] = p_adj
    results_df["Signifikan"] = rej
    
    st.dataframe(results_df.style.format({"Diff": "{:.2f}", "t-Stat": "{:.3f}", "p-Raw": "{:.5f}", "p-Adj (Holm)": "{:.5f}"}))

else:
    st.info("Silakan unggah file CSV Anda untuk memulai.")
