"""
Reproducible analysis for:
"Machine-Learning-Based Predictive Maintenance and Cost-Risk Trade-off in Automated Production Systems"

Protocol (matches the revised manuscript):
  * AI4I 2020 dataset (UCI, doi:10.24432/C5HS5C), raw physical units, NO scaling/normalisation/transformation.
  * Inputs: Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Product type (L/M/H -> 0/1/2),
    engineered load-thermal interaction tau*(Tp - Ta). Target: Machine failure. Failure-mode flags are dropped (leakage).
  * Five learners with library DEFAULT hyperparameters: RF, GBDT, XGBoost, SVM (RBF), MLP.
  * Inverse-frequency class weights w+ = N/(2N+), w- = N/(2N-) passed as sample weights.
  * Stratified 5-fold outer CV; inside each outer training fold an inner stratified 80/20 split:
    model fitted on 80 %, Platt (sigmoid) calibration fitted on the held-out 20 %.
  * Hard labels for Table 4 = calibrated probability >= 0.5.
  * External feature validation: out-of-fold permutation importance (PR-AUC drop) -> ordered sets,
    Kendall's W / Spearman across folds and across learners, ordered-prefix subsets, quintile dose-response
    with Spearman and Jonckheere-Terpstra trend test (tie-corrected normal approximation).
  * Cost-risk evaluation on the out-of-fold calibrated probabilities of the selected model.
"""
import json, time, warnings, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             average_precision_score, brier_score_loss)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "ai4i2020.csv")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- data
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
UCI_MD5 = "f0adaa9c5e25366c6df1cc2684479871"  # MD5 of the original ai4i2020.csv distributed by UCI


def ensure_dataset(path=DATA):
    """Download the unmodified AI4I 2020 file from the UCI repository if it is not present locally."""
    import hashlib, io, urllib.request, zipfile
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Local data not found -> downloading from UCI ({UCI_ZIP_URL})", flush=True)
        req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith("ai4i2020.csv"))
            with open(path, "wb") as fh:
                fh.write(zf.read(name))
        print("Saved", path, flush=True)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if md5 != UCI_MD5:
        print(f"WARNING: MD5 {md5} differs from the original UCI file ({UCI_MD5}); results may not match the paper.")
    else:
        print("Dataset checksum verified (identical to the original UCI file).")


ensure_dataset()
df = pd.read_csv(DATA)
X = pd.DataFrame({
    "Air temperature": df["Air temperature [K]"],
    "Process temperature": df["Process temperature [K]"],
    "Rotational speed": df["Rotational speed [rpm]"],
    "Torque": df["Torque [Nm]"],
    "Tool wear": df["Tool wear [min]"],
    "Product type": df["Type"].map({"L": 0, "M": 1, "H": 2}),
    "Load-thermal interaction": df["Torque [Nm]"] * (df["Process temperature [K]"] - df["Air temperature [K]"]),
})
y = df["Machine failure"].values.astype(int)
FEATS = list(X.columns)
N, Npos = len(y), int(y.sum())
Nneg = N - Npos
W_POS, W_NEG = N / (2 * Npos), N / (2 * Nneg)
print(f"N={N}, failures={Npos}, rate={Npos/N:.4f}, w+={W_POS:.3f}, w-={W_NEG:.3f}")


def make_models():
    return {
        "Random forest": RandomForestClassifier(random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(random_state=SEED, n_jobs=4, verbosity=0),
        "Support vector machine": SVC(random_state=SEED),
        "Neural network": MLPClassifier(random_state=SEED),
    }


class CalibratedModel:
    """Weighted base learner + Platt mapping fitted on an inner held-out split."""

    def __init__(self, base):
        self.base = base

    def raw(self, X):
        if hasattr(self.base, "predict_proba"):
            return self.base.predict_proba(X)[:, 1]
        return self.base.decision_function(X)

    def raw_label(self, X):  # hard label of the class-weighted model (cost-sensitive classifier)
        return self.base.predict(X).astype(int)

    def fit(self, X, y, seed=SEED):
        Xf, Xc, yf, yc = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sw = np.where(yf == 1, W_POS, W_NEG)
        self.base.fit(Xf, yf, sample_weight=sw)
        s = self.raw(Xc).reshape(-1, 1)
        self.cal = LogisticRegression(C=1e6, max_iter=1000).fit(s, yc)
        return self

    def predict_proba(self, X):
        p = self.cal.predict_proba(self.raw(X).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # sklearn compatibility for permutation_importance
    def get_params(self, deep=False):
        return {"base": self.base}

    def score(self, X, y):
        return average_precision_score(y, self.predict_proba(X)[:, 1])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def pr_auc_scorer(est, X, y):
    return average_precision_score(y, est.predict_proba(X)[:, 1])


outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(outer.split(X, y))

# ----------------------------------------------------------------------------- 1) model comparison
metrics = {m: [] for m in make_models()}
oof = {m: np.zeros(N) for m in make_models()}
oof_raw = {m: np.zeros(N) for m in make_models()}
oof_cs = {m: np.zeros(N, dtype=int) for m in make_models()}
imp = {m: [] for m in make_models()}  # per fold permutation importance vectors
calib = {m: [] for m in make_models()}
latency = {}

for k, (tr, te) in enumerate(folds):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    for name, base in make_models().items():
        t0 = time.time()
        cm = CalibratedModel(base).fit(Xtr, ytr, seed=SEED + k)
        p = cm.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        raw = cm.raw(Xte)
        oof[name][te] = p
        oof_raw[name][te] = raw
        oof_cs[name][te] = cm.raw_label(Xte)
        metrics[name].append(dict(
            Accuracy=accuracy_score(yte, yhat),
            Precision=precision_score(yte, yhat, zero_division=0),
            Recall=recall_score(yte, yhat),
            NormalF1=f1_score(yte, yhat, pos_label=0),
            FailureF1=f1_score(yte, yhat, pos_label=1),
            MacroF1=f1_score(yte, yhat, average="macro"),
            PRAUC=average_precision_score(yte, p),
        ))
        rawp = raw if hasattr(base, "predict_proba") else 1 / (1 + np.exp(-raw))
        calib[name].append(dict(brier_raw=brier_score_loss(yte, rawp), brier_cal=brier_score_loss(yte, p),
                                ece_raw=ece(yte, rawp), ece_cal=ece(yte, p)))
        pi = permutation_importance(cm, Xte, yte, scoring=pr_auc_scorer, n_repeats=10,
                                    random_state=SEED + k, n_jobs=1)
        imp[name].append(pi.importances_mean)
        if k == 0:
            t1 = time.perf_counter()
            for _ in range(5):
                cm.predict_proba(Xte)
            latency[name] = (time.perf_counter() - t1) / 5 / len(Xte) * 1e3  # ms / record
        print(f"fold {k} {name:24s} F1(fail)={metrics[name][-1]['FailureF1']:.3f} "
              f"PRAUC={metrics[name][-1]['PRAUC']:.3f} ({time.time()-t0:.1f}s)", flush=True)

table4 = []
for name in metrics:
    d = pd.DataFrame(metrics[name])
    row = {"Model": name}
    for c in d.columns:
        row[c + "_mean"] = d[c].mean()
        row[c + "_sd"] = d[c].std(ddof=1)
    table4.append(row)
table4 = pd.DataFrame(table4).sort_values(["FailureF1_mean", "PRAUC_mean"], ascending=False)
table4.to_csv(os.path.join(OUT, "table4_model_comparison.csv"), index=False)
BEST = table4.iloc[0]["Model"]
print("\nSelected model:", BEST)
print(table4.round(3).to_string())

# ----------------------------------------------------------------------------- 2) ordered-set consistency


def kendall_w(R):
    """R: m raters x n items matrix of ranks. Returns W, chi2, p (no tie correction needed for avg ranks approx)."""
    m, n = R.shape
    Rj = R.sum(axis=0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    # tie correction
    T = 0.0
    for row in R:
        _, c = np.unique(row, return_counts=True)
        T += (c ** 3 - c).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n) - m * T)
    chi2 = m * (n - 1) * W
    p = 1 - stats.chi2.cdf(chi2, n - 1)
    return W, chi2, p


def mean_pairwise_spearman(R):
    m = R.shape[0]
    vals = [stats.spearmanr(R[i], R[j])[0] for i in range(m) for j in range(i + 1, m)]
    return float(np.nanmean(vals))


def ranks_from_imp(v):  # rank 1 = most important
    return stats.rankdata(-np.asarray(v), method="average")


I_best = np.array(imp[BEST])  # folds x feats
R_folds = np.vstack([ranks_from_imp(v) for v in I_best])
consensus_order = [FEATS[i] for i in np.argsort(-I_best.mean(axis=0))]
W_f, chi_f, p_f = kendall_w(R_folds)
rho_f = mean_pairwise_spearman(R_folds)

model_mean_imp = {m: np.array(imp[m]).mean(axis=0) for m in imp}
R_models = np.vstack([ranks_from_imp(model_mean_imp[m]) for m in imp])
W_m, chi_m, p_m = kendall_w(R_models)
rho_m = mean_pairwise_spearman(R_models)

top3 = consensus_order[:3]
tail = consensus_order[3:]
tail_idx = [FEATS.index(f) for f in tail]
R_tail = np.vstack([ranks_from_imp(v[tail_idx]) for v in I_best])
W_t, chi_t, p_t = kendall_w(R_tail)

top3_idx = [FEATS.index(f) for f in top3]
top3_invariant_folds = all(list(np.argsort(-v)[:3]) == top3_idx for v in I_best)
top3_by_model = {m: [FEATS[i] for i in np.argsort(-model_mean_imp[m])[:3]] for m in imp}
top3_by_model_fold = {m: [[FEATS[i] for i in np.argsort(-v)[:3]] for v in imp[m]] for m in imp}

# ordered-prefix subsets with the selected learner
prefix = []
for k_ in range(1, len(consensus_order) + 1):
    cols = consensus_order[:k_]
    f1s, aps = [], []
    for k, (tr, te) in enumerate(folds):
        cm = CalibratedModel(make_models()[BEST]).fit(X.iloc[tr][cols], y[tr], seed=SEED + k)
        p = cm.predict_proba(X.iloc[te][cols])[:, 1]
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int)))
        aps.append(average_precision_score(y[te], p))
    prefix.append(dict(k=k_, features=cols, F1_mean=np.mean(f1s), F1_sd=np.std(f1s, ddof=1),
                       PRAUC_mean=np.mean(aps), PRAUC_sd=np.std(aps, ddof=1)))
    print("prefix", k_, cols[-1], round(np.mean(f1s), 3), round(np.mean(aps), 3), flush=True)

# ----------------------------------------------------------------------------- 3) dose-response


def jonckheere(groups):
    """Tie-corrected Jonckheere-Terpstra test (normal approximation). groups: list of 1-D arrays ordered."""
    k = len(groups)
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            a, b = np.asarray(groups[i]), np.asarray(groups[j])
            bs = np.sort(b)
            less = np.searchsorted(bs, a, side="right")  # count b <= a
            leq = np.searchsorted(bs, a, side="left")    # count b < a
            gt = len(b) - less
            ties = less - leq
            J += gt.sum() + 0.5 * ties.sum()
    n = np.array([len(g) for g in groups], dtype=float)
    Nn = n.sum()
    allv = np.concatenate(groups)
    _, t = np.unique(allv, return_counts=True)
    t = t.astype(float)
    EJ = (Nn ** 2 - (n ** 2).sum()) / 4
    V = ((Nn * (Nn - 1) * (2 * Nn + 5) - (n * (n - 1) * (2 * n + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72
         + ((n * (n - 1) * (n - 2)).sum() * (t * (t - 1) * (t - 2)).sum()) / (36 * Nn * (Nn - 1) * (Nn - 2))
         + ((n * (n - 1)).sum() * (t * (t - 1)).sum()) / (8 * Nn * (Nn - 1)))
    z = (J - EJ) / np.sqrt(V)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


pbest = oof[BEST]
dose = {}
for f in ["Torque", "Tool wear", "Load-thermal interaction", "Rotational speed", "Process temperature",
          "Air temperature"]:
    q = pd.qcut(X[f], 5, labels=False, duplicates="drop")
    edges = pd.qcut(X[f], 5, retbins=True, duplicates="drop")[1]
    rows = []
    groups_y, groups_p = [], []
    for g in sorted(q.unique()):
        m = (q == g).values
        rows.append(dict(Q=int(g) + 1, lo=float(X[f][m].min()), hi=float(X[f][m].max()), n=int(m.sum()),
                         failures=int(y[m].sum()), rate=float(y[m].mean()), mean_p=float(pbest[m].mean())))
        groups_y.append(y[m])
        groups_p.append(pbest[m])
    rates = [r["rate"] for r in rows]
    mps = [r["mean_p"] for r in rows]
    zy, py_ = jonckheere(groups_y)
    zp, pp_ = jonckheere(groups_p)
    dose[f] = dict(quintiles=rows, edges=[float(e) for e in edges],
                   spearman_rate=float(stats.spearmanr(range(len(rates)), rates)[0]),
                   spearman_meanp=float(stats.spearmanr(range(len(mps)), mps)[0]),
                   monotone_rate=bool(np.all(np.diff(rates) > 0)),
                   monotone_meanp=bool(np.all(np.diff(mps) > 0)),
                   JT_z_y=zy, JT_p_y=py_, JT_z_p=zp, JT_p_p=pp_)
dose_extra = {
    "Torque>55": dict(n=int((X["Torque"] > 55).sum()), rate=float(y[(X["Torque"] > 55).values].mean())),
    "Tool wear>200": dict(n=int((X["Tool wear"] > 200).sum()), rate=float(y[(X["Tool wear"] > 200).values].mean())),
}

# ----------------------------------------------------------------------------- 4) cost-risk evaluation
RHO = 0.9


def policy_actions(p, cs_label, cp, cf, rho=RHO):
    tau = cp / (rho * cf)
    return {
        "Run-to-failure": np.zeros(N),
        "Periodic PM": np.full(N, 0.8),  # services 80 % of operating states (expected value)
        "Fixed-threshold PdM (threshold = 0.5)": (p >= 0.5).astype(float),
        "Cost-sensitive classifier": cs_label.astype(float),
        "Cost-adaptive (proposed)": (p >= tau).astype(float),
    }


def evaluate(a, yy, cp, cf, rho=RHO, norm=None):
    planned = (a * cp).sum()
    risk = ((1 - a * rho) * yy * cf).sum()
    norm = norm if norm is not None else (yy * cf).sum()
    return dict(planned=100 * planned / norm, risk=100 * risk / norm, total=100 * (planned + risk) / norm,
                rate=a.mean(), intercepted=float((a * yy).sum() / yy.sum()))


cs = oof_cs[BEST]
CP, CF = 1.0, 20.0
acts = policy_actions(pbest, cs, CP, CF)
table5 = {k: evaluate(v, y, CP, CF) for k, v in acts.items()}
print("\nTable 5"); print(pd.DataFrame(table5).T.round(3))

# bootstrap
rng = np.random.default_rng(SEED)
B = 2000
diff_cs, diff_fx = [], []
a_p, a_c, a_f = acts["Cost-adaptive (proposed)"], acts["Cost-sensitive classifier"], acts["Fixed-threshold PdM (threshold = 0.5)"]
for _ in range(B):
    idx = rng.integers(0, N, N)
    yy = y[idx]
    nrm = (yy * CF).sum()
    tp = evaluate(a_p[idx], yy, CP, CF, norm=nrm)["total"]
    diff_cs.append(evaluate(a_c[idx], yy, CP, CF, norm=nrm)["total"] - tp)
    diff_fx.append(evaluate(a_f[idx], yy, CP, CF, norm=nrm)["total"] - tp)
diff_cs, diff_fx = np.array(diff_cs), np.array(diff_fx)


def boot_summary(d):
    frac = float((d > 0).mean())
    p_two = 2 * min(frac, 1 - frac)
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                frac_positive=frac, p_two_sided=max(p_two, 1 / B))


boot = {"vs_cost_sensitive": boot_summary(diff_cs), "vs_fixed_threshold": boot_summary(diff_fx)}

# frontier (Fig. 2)
frontier = []
for ratio in [5, 7.5, 10, 15, 20, 25, 30, 35, 40]:
    a = (pbest >= CP / (RHO * ratio)).astype(float)
    e = evaluate(a, y, CP, ratio)
    e.update(ratio=ratio, threshold=CP / (RHO * ratio))
    frontier.append(e)

# sensitivity (Fig. 4)
sens = []
rng2 = np.random.default_rng(SEED + 1)
raw_cs = oof_raw[BEST]
for sigma in [0.0, 0.02, 0.05, 0.10]:
    noise = rng2.normal(0, sigma, N) if sigma > 0 else np.zeros(N)
    pn = np.clip(pbest + noise, 0, 1)
    # same perturbation applied to the class-weighted model's score before its own 0.5 cut-off
    rawn = raw_cs if BEST == "Support vector machine" else np.clip(raw_cs + noise, 0, 1)
    csn = (rawn >= (0 if BEST == "Support vector machine" else 0.5)).astype(float)
    for ratio in [2, 3, 5, 7.5, 10, 15, 20, 25, 30, 35, 40]:
        tp = evaluate((pn >= CP / (RHO * ratio)).astype(float), y, CP, ratio)["total"]
        tc = evaluate(csn, y, CP, ratio)["total"]
        tf = evaluate((pn >= 0.5).astype(float), y, CP, ratio)["total"]
        sens.append(dict(sigma=sigma, ratio=ratio, adv_vs_cs=100 * (tc - tp) / tc, adv_vs_fixed=100 * (tf - tp) / tf,
                         total_prop=tp, total_cs=tc))
sens = pd.DataFrame(sens)
sens["noise"] = "additive"
# logit-scale (multiplicative-odds) perturbation applied identically to both probability streams
def logit(v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))
sens2 = []
rng3 = np.random.default_rng(SEED + 2)
for s_ in [0.0, 0.25, 0.5, 1.0]:
    noise = rng3.normal(0, s_, N) if s_ > 0 else np.zeros(N)
    pn = 1 / (1 + np.exp(-(logit(pbest) + noise)))
    rn = 1 / (1 + np.exp(-(logit(raw_cs) + noise)))
    csn = (rn >= 0.5).astype(float)
    for ratio in [2, 3, 5, 7.5, 10, 15, 20, 25, 30, 35, 40]:
        tp = evaluate((pn >= CP / (RHO * ratio)).astype(float), y, CP, ratio)["total"]
        tc = evaluate(csn, y, CP, ratio)["total"]
        tf = evaluate((pn >= 0.5).astype(float), y, CP, ratio)["total"]
        sens2.append(dict(sigma=s_, ratio=ratio, adv_vs_cs=100 * (tc - tp) / tc, adv_vs_fixed=100 * (tf - tp) / tf,
                          total_prop=tp, total_cs=tc, noise="logit"))
sens = pd.concat([sens, pd.DataFrame(sens2)], ignore_index=True)
sens.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
np.save(os.path.join(OUT, "oof_raw_selected.npy"), raw_cs)
np.save(os.path.join(OUT, "oof_costsensitive_labels.npy"), cs)
np.save(os.path.join(OUT, "y.npy"), y)
pd.DataFrame({m: oof[m] for m in oof}).to_csv(os.path.join(OUT, "oof_calibrated_probabilities_all_models.csv"), index=False)
pd.DataFrame(prefix).to_csv(os.path.join(OUT, "ordered_prefix_subsets.csv"), index=False)
pd.DataFrame([dict(feature=f, **q_) for f in dose for q_ in dose[f]["quintiles"]]).to_csv(
    os.path.join(OUT, "dose_response_quintiles.csv"), index=False)
pd.DataFrame(table5).T.to_csv(os.path.join(OUT, "table5_economic_comparison.csv"))
pd.DataFrame(frontier).to_csv(os.path.join(OUT, "frontier.csv"), index=False)
pd.DataFrame({m: np.array(imp[m]).mean(0) for m in imp}, index=FEATS).to_csv(
    os.path.join(OUT, "permutation_importance_by_model.csv"))
pd.DataFrame(I_best, columns=FEATS).to_csv(os.path.join(OUT, "permutation_importance_selected_by_fold.csv"),
                                           index_label="fold")

# ----------------------------------------------------------------------------- save
cal = pd.DataFrame(calib[BEST]).mean().to_dict()
res = dict(
    N=N, failures=Npos, w_pos=W_POS, w_neg=W_NEG, selected=BEST, features=FEATS,
    table4=table4.to_dict(orient="records"),
    calibration_selected=cal, latency_ms_per_record=latency,
    consistency=dict(consensus_order=consensus_order,
                     mean_imp_best=dict(zip(FEATS, I_best.mean(0).tolist())),
                     sd_imp_best=dict(zip(FEATS, I_best.std(0, ddof=1).tolist())),
                     W_folds=W_f, chi2_folds=chi_f, p_folds=p_f, spearman_folds=rho_f,
                     W_models=W_m, chi2_models=chi_m, p_models=p_m, spearman_models=rho_m,
                     W_tail=W_t, chi2_tail=chi_t, p_tail=p_t,
                     top3_invariant_folds=top3_invariant_folds, top3_by_model=top3_by_model,
                     top3_by_model_fold=top3_by_model_fold,
                     model_mean_imp={m: dict(zip(FEATS, v.tolist())) for m, v in model_mean_imp.items()},
                     prefix=prefix),
    dose=dose, dose_extra=dose_extra,
    table5=table5, bootstrap=boot, frontier=frontier,
    sensitivity_min_adv_vs_cs=float(sens["adv_vs_cs"].min()),
    mechanism_reference="AI4I 2020 generating rules (Matzka, 2020): TWF tool wear 200-240 min; HDF (Tp-Ta)<8.6 K and "
                        "speed<1380 rpm; PWF power=torque*omega outside [3500, 9000] W; OSF tool wear*torque > "
                        "11000/12000/13000 min.Nm for L/M/H; RNF 0.1 % random.",
)
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, default=float)
np.save(os.path.join(OUT, "oof_prob_selected.npy"), pbest)
print("\nSaved results to", OUT)
