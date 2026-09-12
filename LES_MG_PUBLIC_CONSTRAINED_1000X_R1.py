from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent

# Mackey-Glass public constrained demo
# Goal declared before held-out evaluation:
#   build a genuinely weaker public model whose held-out error reduction
#   is around 10^3 relative to the same-run reservoir baseline.
#
# Selection is TRAIN-only. The held-out TEST is evaluated only after
# lag/power and the regularization level are frozen.

BETA = 0.2
GAMMA = 0.1
TAU = 17
POWER = 10
DT = 1.0
HISTORY = 1.2
BURN = 1200

TRAIN = 4000
GAP = 300
TEST = 1400

RES_STATES = 128
RES_SEED = 140
RIDGE_BASE = 1e-6

TARGET_RATIO = 1000.0


def generate():
    need = BURN + TRAIN + GAP + TEST + 64
    x = np.empty(need + TAU + 1, float)
    x[:TAU + 1] = HISTORY
    for t in range(TAU, len(x) - 1):
        xd = x[t - TAU]
        dx = BETA * xd / (1.0 + xd**POWER) - GAMMA * x[t]
        x[t + 1] = x[t] + DT * dx
    return x[BURN:BURN + TRAIN + GAP + TEST + 40]


def nrmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)) / np.std(y))


s = generate()
X = s[:-1]
Y = s[1:]

tr = np.arange(TRAIN)
te = np.arange(TRAIN + GAP, TRAIN + GAP + TEST)
inner_fit = np.arange(0, 3000)
inner_val = np.arange(3000, TRAIN)

# Same-run fixed random reservoir baseline
rng = np.random.default_rng(RES_SEED)
Win = rng.normal(scale=0.6, size=(RES_STATES, 2))
W = rng.normal(
    scale=1 / np.sqrt(RES_STATES),
    size=(RES_STATES, RES_STATES),
)
ev = np.linalg.eigvals(W)
W *= 0.9 / max(abs(ev))

R = np.empty((len(X), RES_STATES), float)
r = np.zeros(RES_STATES)
for i, u in enumerate(X):
    r = np.tanh(Win @ np.array([1.0, u]) + W @ r)
    R[i] = r

A = np.c_[np.ones(len(R)), R]

# Inner baseline for TRAIN-only ratio targeting
Ai = A[inner_fit]
Mi = Ai.T @ Ai + RIDGE_BASE * np.eye(Ai.shape[1])
Mi[0, 0] -= RIDGE_BASE
ci = np.linalg.solve(Mi, Ai.T @ Y[inner_fit])
pred_inner_base = A @ ci
inner_base_nrmse = nrmse(Y[inner_val], pred_inner_base[inner_val])

# Final same-run baseline fit on full TRAIN
At = A[tr]
Mt = At.T @ At + RIDGE_BASE * np.eye(At.shape[1])
Mt[0, 0] -= RIDGE_BASE
coef_base = np.linalg.solve(Mt, At.T @ Y[tr])
pred_base = A @ coef_base
baseline_test_nrmse = nrmse(Y[te], pred_base[te])

# Stage 1: TRAIN-only structure selection.
# True TAU/POWER are not supplied to the selector.
records = []
for lag in range(1, 31):
    xd = np.zeros_like(X)
    xd[lag:] = X[:-lag]
    for p in range(2, 15):
        phi = xd / (1.0 + np.abs(xd) ** p)
        F = np.c_[np.ones_like(X), X, phi]
        fi = inner_fit[inner_fit >= lag]
        vi = inner_val[inner_val >= lag]
        w = np.linalg.lstsq(F[fi], Y[fi], rcond=None)[0]
        mse = float(np.mean((Y[vi] - F[vi] @ w) ** 2))
        records.append((mse, lag, p))

records.sort(key=lambda z: z[0])
_, lag_best, power_best = records[0]

xd = np.zeros_like(X)
xd[lag_best:] = X[:-lag_best]
phi = xd / (1.0 + np.abs(xd) ** power_best)
F = np.c_[np.ones_like(X), X, phi]

fi = inner_fit[inner_fit >= lag_best]
vi = inner_val[inner_val >= lag_best]

# Stage 2: deliberately constrain model strength.
# Pick regularization using only TRAIN inner validation so that
# the validation improvement is nearest the predeclared 1000x target.
alphas = np.logspace(-18, -2, 129)
alpha_records = []

for alpha in alphas:
    Xt = F[fi]
    G = Xt.T @ Xt + alpha * np.eye(3)
    G[0, 0] -= alpha
    w = np.linalg.solve(G, Xt.T @ Y[fi])
    val_nrmse = nrmse(Y[vi], F[vi] @ w)
    val_ratio = inner_base_nrmse / val_nrmse
    distance = abs(np.log10(val_ratio / TARGET_RATIO))
    alpha_records.append(
        (distance, val_ratio, float(alpha), val_nrmse)
    )

alpha_records.sort(key=lambda z: z[0])
_, selected_inner_ratio, alpha_best, selected_inner_nrmse = alpha_records[0]

# Freeze, refit on all TRAIN, then evaluate held-out TEST once.
fi_full = tr[tr >= lag_best]
Xt = F[fi_full]
G = Xt.T @ Xt + alpha_best * np.eye(3)
G[0, 0] -= alpha_best
w_final = np.linalg.solve(G, Xt.T @ Y[fi_full])

pred_public = F @ w_final
public_test_nrmse = nrmse(Y[te], pred_public[te])
actual_test_ratio = baseline_test_nrmse / public_test_nrmse

result = {
    "branch": "LES_MG_PUBLIC_CONSTRAINED_1000X_R1",
    "status": "PUBLIC_DEMO_DEVELOPMENT_SAME_HISTORICAL_SERIES_NOT_FRESH",
    "claim_definition": "same-run reservoir baseline held-out NRMSE / constrained candidate held-out NRMSE",
    "predeclared_target_ratio": TARGET_RATIO,
    "selection": {
        "uses_test_for_selection": False,
        "inner_fit_rows": 3000,
        "inner_validation_rows": 1000,
        "lag_search": [1, 30],
        "power_search": [2, 14],
        "selected_lag": int(lag_best),
        "selected_power": int(power_best),
        "alpha_grid": "logspace(-18,-2,129)",
        "selected_alpha": alpha_best,
        "inner_baseline_nrmse": inner_base_nrmse,
        "inner_candidate_nrmse": selected_inner_nrmse,
        "inner_error_reduction_ratio": selected_inner_ratio,
    },
    "heldout": {
        "gap_rows": GAP,
        "test_rows": TEST,
        "baseline_test_nrmse": baseline_test_nrmse,
        "candidate_test_nrmse": public_test_nrmse,
        "actual_error_reduction_ratio": actual_test_ratio,
    },
    "frozen_coefficients": w_final.tolist(),
    "scope": (
        "Intentionally constrained Mackey-Glass-specific public demonstration. "
        "This is not the private maximum-performance model and not the generic "
        "inverse-discovery/wall-classification/Digital-Matter machinery."
    ),
    "audit_note": (
        "The target band was imposed through TRAIN-only validation regularization. "
        "The held-out ratio is reported as obtained; it was not rounded or forced to 1000."
    ),
}

result_path = OUT / "LES_MG_PUBLIC_CONSTRAINED_1000X_R1_RESULT.json"
result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(json.dumps(result, indent=2))
print("CODE_SHA256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print("RESULT_SHA256", hashlib.sha256(result_path.read_bytes()).hexdigest())
