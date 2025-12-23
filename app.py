import io 
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

PATH = "/content/Data_HargaBeras.csv"
ALPHA = 0.05
KEEP = ["premium", "medium", "pecah"]

def keputusan_normal(p, alpha=0.05):
    return "NORMAL" if p >= alpha else "TIDAK NORMAL"

def keputusan_homogen(p, alpha=0.05):
    return "HOMOGEN" if p >= alpha else "TIDAK HOMOGEN"

# ============================================================
# 1) DATA PREPARATION (CSV header tidak rapi -> wide -> long)
# ============================================================
raw = pd.read_csv(PATH, header=None)

idx_bulan = raw.apply(
    lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(),
    axis=1
).idxmax()

bulan = raw.loc[idx_bulan, 1:12].tolist()
if len(bulan) != 12:
    raise ValueError(f"Gagal deteksi 12 bulan Jan–Des. Dapat: {len(bulan)} -> {bulan}")

wide = raw.loc[idx_bulan + 1:, [0] + list(range(1, 13))].copy()
wide.columns = ["kualitas"] + bulan
wide["kualitas"] = wide["kualitas"].astype(str).str.strip()

wide = wide[wide["kualitas"].str.lower().isin(KEEP)].copy()

for b in bulan:
    wide[b] = pd.to_numeric(wide[b], errors="coerce")

long = wide.melt(
    id_vars="kualitas",
    value_vars=bulan,
    var_name="bulan",
    value_name="harga"
).dropna(subset=["harga"]).copy()

long["bulan"] = pd.Categorical(long["bulan"], categories=bulan, ordered=True)
long["kualitas"] = long["kualitas"].str.strip().str.title()  # Premium/Medium/Pecah
long = long.sort_values(["bulan", "kualitas"]).reset_index(drop=True)

# >>> Z-SCORE DIHAPUS (tidak ada transformasi)
# mu = long["harga"].mean()
# sd = long["harga"].std(ddof=0)
# long["harga_norm"] = (long["harga"] - mu) / sd


# ============================================================
# 2) DATA CHECK
# ============================================================
print("=== DATA CHECK ===")
print("Bulan:", bulan)
print("Wide shape:", wide.shape, "| Long shape:", long.shape)
print("Kualitas:", sorted(long["kualitas"].unique().tolist()))
print("Jumlah data per kualitas:", long.groupby("kualitas")["harga"].size().to_dict())
print("Missing per kolom (long):")
print(long.isna().sum())
dup = long.duplicated(subset=["bulan", "kualitas"]).sum()
print("Duplikat (bulan,kualitas):", dup)
print("\nPreview long:")
print(long.head(12))


# ============================================================
# 3) EDA (deskriptif + plot)
# ============================================================
print("\n=== EDA (Deskriptif harga) ===")
desc = long.groupby("kualitas")["harga"].agg(["count", "mean", "std", "min", "max"])
print(desc)

pivot = long.pivot_table(index="bulan", columns="kualitas", values="harga").loc[bulan]
plt.figure()
for col in pivot.columns:
    plt.plot(pivot.index.astype(str), pivot[col].values, marker="o", label=col)
plt.xticks(rotation=45)
plt.title("Trend Harga Beras (Jan–Des) per Kualitas")
plt.xlabel("Bulan")
plt.ylabel("Harga")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
kvals = sorted(long["kualitas"].unique())
data = [long.loc[long["kualitas"] == k, "harga"].values for k in kvals]
plt.boxplot(data, labels=kvals)
plt.title("Boxplot Harga per Kualitas")
plt.xlabel("Kualitas")
plt.ylabel("Harga")
plt.tight_layout()
plt.show()


# ============================================================
# 4) NORMALITAS DATA (per populasi) — kalau dosen minta “data normal”
# ============================================================
print("\n=== NORMALITAS DATA per POPULASI (Shapiro, HARGA mentah) ===")
for k, g in long.groupby("kualitas"):
    p = stats.shapiro(g["harga"].to_numpy()).pvalue
    print(f"{k:8s} | p-value={p:.6g} | {keputusan_normal(p, ALPHA)}")


# ============================================================
# 5) HOMOGENITAS + NORMALITAS RESIDUAL (asumsi ANOVA)
#    (model di-fit dulu agar ada residual, tapi ANOVA-nya BELUM ditampilkan)
# ============================================================
# >>> MODEL DIGANTI: pakai HARGA asli (tanpa z-score)
model = smf.ols("harga ~ C(kualitas) + C(bulan)", data=long).fit()

infl = model.get_influence()
resid_stud = infl.resid_studentized_internal

print("\n=== HOMOGENITAS VARIANSI antar kualitas (Levene/Fligner pada RESIDUAL) ===")
tmp = long.copy()
tmp["resid_stud"] = resid_stud
groups = [g["resid_stud"].to_numpy() for _, g in tmp.groupby("kualitas")]

p_lev = stats.levene(*groups, center="median").pvalue
p_flg = stats.fligner(*groups).pvalue
print(f"Levene (median) p-value={p_lev:.6g} | {keputusan_homogen(p_lev, ALPHA)}")
print(f"Fligner         p-value={p_flg:.6g} | {keputusan_homogen(p_flg, ALPHA)}")

print("\n=== NORMALITAS RESIDUAL MODEL (Shapiro) ===")
p_sh = stats.shapiro(resid_stud).pvalue
print(f"Shapiro residual p-value={p_sh:.6g} | {keputusan_normal(p_sh, ALPHA)}")

sm.qqplot(resid_stud, line="45")
plt.title("Q-Q Plot Studentized Residual")
plt.tight_layout()
plt.show()


# ============================================================
# 6) BARU tampilkan ANOVA
# ============================================================
anova_tbl = sm.stats.anova_lm(model, typ=2)
print("\n=== ANOVA (harga ~ kualitas + bulan) ===")
print(anova_tbl)


# ============================================================
# 7) POST-HOC kualitas (model-based; kontrol bulan) + Holm
# ============================================================
names = model.params.index.tolist()
k = len(names)

def ttest(weights):
    w = np.zeros(k)
    for nm, val in weights.items():
        w[names.index(nm)] = val
    r = model.t_test(w)
    return float(r.effect), float(r.tvalue), float(r.pvalue)

tests = [
    ("Pecah - Medium",   {"C(kualitas)[T.Pecah]": 1.0}),
    ("Premium - Medium", {"C(kualitas)[T.Premium]": 1.0}),
    ("Premium - Pecah",  {"C(kualitas)[T.Premium]": 1.0, "C(kualitas)[T.Pecah]": -1.0}),
]

rows, pvals = [], []
for label, wd in tests:
    eff, t, p = ttest(wd)
    rows.append([label, eff, t, p])
    pvals.append(p)

rej, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")

print("\n=== POST-HOC KUALITAS (kontrol bulan) + Holm ===")
for (label, eff, t, p), pa, r in zip(rows, p_adj, rej):
    print(f"{label:16s} | diff={eff:.4f} | t={t:.3f} | p={p:.6g} | p_adj={pa:.6g} | signif={r}")


# ============================================================
# 8) EFFECT SIZE + DIAGNOSTICS
# ============================================================
ss_res = anova_tbl.loc["Residual", "sum_sq"]
eta_kualitas = anova_tbl.loc["C(kualitas)", "sum_sq"] / (anova_tbl.loc["C(kualitas)", "sum_sq"] + ss_res)
eta_bulan = anova_tbl.loc["C(bulan)", "sum_sq"] / (anova_tbl.loc["C(bulan)", "sum_sq"] + ss_res)

print("\n=== EFFECT SIZE (Partial Eta Squared) ===")
print(f"partial eta^2 kualitas = {eta_kualitas:.4f}")
print(f"partial eta^2 bulan    = {eta_bulan:.4f}")

fitted = model.fittedvalues.to_numpy()
plt.figure()
plt.scatter(fitted, resid_stud)
plt.axhline(0)
plt.title("Residual vs Fitted")
plt.xlabel("Fitted")
plt.ylabel("Studentized Residual")
plt.tight_layout()
plt.show()
