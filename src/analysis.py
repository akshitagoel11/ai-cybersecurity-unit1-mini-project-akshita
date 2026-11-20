"""Tasks 2-5: EDA, ML classification, MLP comparison, ethics experiments.
Saves figures to figures/ and all numbers to results.json for the report."""
import json, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)
SEED = 231
R = {}
df = pd.read_csv("data/cyber_events.csv")
CAT = ["protocol", "source_country", "device_type"]
NUM = ["login_hour", "failed_logins", "session_duration_s", "bytes_in",
       "bytes_out", "ports_accessed", "password_age_days", "mfa_enabled"]
Y = "is_suspicious"
lab = df[Y].map({0: "Normal", 1: "Suspicious"})
BLUE, RED = "#2F6DA3", "#C8473F"

# ---------------- Task 2: EDA ----------------
R["n"] = len(df); R["n_susp"] = int(df[Y].sum()); R["susp_pct"] = round(100*df[Y].mean(), 1)
R["missing_pct"] = {c: round(100*df[c].isna().mean(), 1) for c in df.columns if df[c].isna().any()}
by_c = (df.groupby("source_country")[Y].mean()*100).sort_values(ascending=False)
by_p = (df.groupby("protocol")[Y].mean()*100).sort_values(ascending=False)
R["by_country"] = by_c.round(1).to_dict(); R["by_protocol"] = by_p.round(1).to_dict()
R["mfa_rate"] = (df.groupby("mfa_enabled")[Y].mean()*100).round(1).to_dict()
corr = df[NUM+[Y]].corr()
R["corr_label"] = corr[Y].drop(Y).round(3).sort_values(key=abs, ascending=False).to_dict()

fig, ax = plt.subplots(figsize=(4.2, 3))
sns.countplot(x=lab, order=["Normal", "Suspicious"], palette=[BLUE, RED], ax=ax)
for p in ax.patches: ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax.set_title("Normal vs Suspicious Events"); ax.set_xlabel(""); fig.tight_layout(); fig.savefig("figures/fig1_class_balance.png", dpi=170); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, s, t in [(axs[0], by_c, "source country"), (axs[1], by_p, "protocol")]:
    sns.barplot(x=s.index, y=s.values, color=RED, ax=ax)
    ax.set_ylabel("% suspicious"); ax.set_xlabel(""); ax.set_title(f"Suspicious rate by {t}")
    ax.tick_params(axis="x", rotation=40)
fig.tight_layout(); fig.savefig("figures/fig2_rate_country_protocol.png", dpi=170); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
pal = {"Normal": BLUE, "Suspicious": RED}
sns.histplot(df, x="failed_logins", hue=lab, multiple="dodge", discrete=True, palette=pal, ax=axs[0]); axs[0].set_title("Failed logins")
sns.histplot(df, x="login_hour", hue=lab, bins=24, palette=pal, ax=axs[1]); axs[1].set_title("Login hour")
sns.histplot(df, x="password_age_days", hue=lab, bins=20, palette=pal, ax=axs[2]); axs[2].set_title("Password age (days)")
fig.tight_layout(); fig.savefig("figures/fig3_distributions.png", dpi=170); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
for ax, c in zip(axs, ["bytes_out", "ports_accessed", "session_duration_s"]):
    sns.boxplot(x=lab, y=df[c], order=["Normal", "Suspicious"], palette=[BLUE, RED], ax=ax); ax.set_title(c); ax.set_xlabel("")
    if c == "bytes_out": ax.set_yscale("log")
fig.tight_layout(); fig.savefig("figures/fig4_boxplots.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(6.2, 4.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation Heatmap"); fig.tight_layout(); fig.savefig("figures/fig5_heatmap.png", dpi=170); plt.close()

# ---------------- Task 3: preprocessing + ML ----------------
X = df[CAT+NUM]; y = df[Y]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True,
                                          max_iter=300, random_state=SEED),
}
fitted, R["models"], cms = {}, {}, {}
for name, m in models.items():
    pipe = Pipeline([("prep", prep()), ("clf", m)])
    t0 = time.perf_counter(); pipe.fit(Xtr, ytr); tt = time.perf_counter()-t0
    t0 = time.perf_counter(); pred = pipe.predict(Xte); pt = time.perf_counter()-t0
    fitted[name] = pipe
    cms[name] = confusion_matrix(yte, pred)
    R["models"][name] = dict(accuracy=round(accuracy_score(yte, pred), 4),
        precision=round(precision_score(yte, pred), 4), recall=round(recall_score(yte, pred), 4),
        f1=round(f1_score(yte, pred), 4), train_s=round(tt, 4), predict_s=round(pt, 4),
        cm=cms[name].tolist())
R["test_n"] = len(yte); R["test_susp"] = int(yte.sum())

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
for ax, (n, cm) in zip(axs, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "Susp."], yticklabels=["Normal", "Susp."])
    ax.set_title(n); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.tight_layout(); fig.savefig("figures/fig6_confusion.png", dpi=170); plt.close()

# ---------------- Task 4: ML vs DL ----------------
names = list(models); mets = ["accuracy", "precision", "recall", "f1"]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3))
w = 0.2
for i, mt in enumerate(mets):
    axs[0].bar(np.arange(3)+i*w, [R["models"][n][mt] for n in names], w, label=mt)
axs[0].set_xticks(np.arange(3)+1.5*w); axs[0].set_xticklabels(["LogReg", "Tree", "MLP"])
axs[0].set_ylim(0, 1.05); axs[0].legend(ncol=2, fontsize=7); axs[0].set_title("Detection performance")
axs[1].bar(["LogReg", "Tree", "MLP"], [R["models"][n]["train_s"] for n in names], color=[BLUE, "#E0A030", RED])
axs[1].set_title("Training time (seconds)")
fig.tight_layout(); fig.savefig("figures/fig7_ml_vs_dl.png", dpi=170); plt.close()

# ---------------- Task 5: ethics ----------------
# 5a threshold sweep on the MLP
proba = fitted["Neural Network (MLP)"].predict_proba(Xte)[:, 1]
rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (proba >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    rows.append(dict(t=t, fp=int(fp), fn=int(fn), fpr=round(100*fp/(fp+tn), 1),
                     recall=round(100*tp/(tp+fn), 1), precision=round(100*tp/max(tp+fp, 1), 1)))
R["threshold"] = rows
tt = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(tt.t, tt.fpr, "o-", color=RED, label="False-positive rate (%)")
ax.plot(tt.t, tt.recall, "s-", color=BLUE, label="Recall (%)")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("%"); ax.set_title("Threshold trade-off"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig8_threshold.png", dpi=170); plt.close()

# 5b biased labels: legit non-IN users in training wrongly relabelled suspicious
rng = np.random.default_rng(SEED)
ytr_b = ytr.copy()
foreign_legit = (Xtr.source_country != "IN") & (ytr == 0) & (~Xtr.source_country.isin(["RU", "CN", "UNKNOWN"]))
flip = foreign_legit & (rng.random(len(ytr)) < 0.5)
ytr_b[flip] = 1
R["flipped"] = int(flip.sum())
def group_fpr(pipe):
    pred = pipe.predict(Xte); out = {}
    legit = yte == 0
    for g, mask in {"India (IN)": Xte.source_country == "IN",
                    "Foreign (US/UK/DE/FR)": Xte.source_country.isin(["US", "UK", "DE", "FR"])}.items():
        m = legit & mask; out[g] = round(100*pred[m].mean(), 2)
    return out
clean = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
biased = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr_b)
R["fpr_clean"] = group_fpr(clean); R["fpr_biased"] = group_fpr(biased)
fig, ax = plt.subplots(figsize=(5, 3.2))
gs = list(R["fpr_clean"]); x = np.arange(2)
ax.bar(x-0.18, [R["fpr_clean"][g] for g in gs], 0.36, color=BLUE, label="Trained on clean labels")
ax.bar(x+0.18, [R["fpr_biased"][g] for g in gs], 0.36, color=RED, label="Trained on biased labels")
ax.set_xticks(x); ax.set_xticklabels(gs); ax.set_ylabel("FPR (%)"); ax.legend(fontsize=7)
ax.set_title("False-positive rate by group"); fig.tight_layout(); fig.savefig("figures/fig9_bias.png", dpi=170); plt.close()

# 5c explainability: LR coefficients vs MLP permutation importance
lr = fitted["Logistic Regression"]
fn_ = lr.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[f.split("__")[1] for f in fn_])
top = coef.reindex(coef.abs().sort_values(ascending=False).head(10).index).sort_values()
R["top_coef"] = top.round(2).to_dict()
pi = permutation_importance(fitted["Neural Network (MLP)"], Xte, yte, n_repeats=5, random_state=SEED, scoring="f1")
pis = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
R["perm_imp"] = pis.round(3).to_dict()
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
top.plot.barh(ax=axs[0], color=BLUE); axs[0].set_title("Logistic Regression: top coefficients")
pis.head(8)[::-1].plot.barh(ax=axs[1], color="#E0A030"); axs[1].set_title("MLP: permutation importance (approx.)")
fig.tight_layout(); fig.savefig("figures/fig10_explain.png", dpi=170); plt.close()

json.dump(R, open("results.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in ["susp_pct", "models", "by_country", "by_protocol", "corr_label", "fpr_clean", "fpr_biased", "flipped", "threshold"]}, indent=1)[:4500])
