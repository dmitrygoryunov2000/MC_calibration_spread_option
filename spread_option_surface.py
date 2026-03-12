"""
Spread Option Surface  –  Kirk's Approximation  vs  Monte Carlo
================================================================
Assets  : S1 = S2 = 100,  σ1 = σ2 (flat),  ρ = 90%
Tenors  : 0.5 Y, 1.0 Y, 1.5 Y
Strikes : 90 %, 100 %, 110 % of S1  →  K_spread = S1·k – S2  =  –10, 0, +10
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend – saves to file without blocking
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (side-effect register)
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
S1        = 100.0          # spot price asset 1
S2        = 100.0          # spot price asset 2
RHO       = 0.90           # correlation
SIGMA     = 0.20           # flat vol – same for both assets
R         = 0.00           # risk-free rate (zero for clarity)
N_SIMS    = 300_000        # Monte Carlo paths

TENORS       = np.array([0.5, 1.0, 1.5])
STRIKE_PCTS  = np.array([0.90, 1.00, 1.10])            # as fraction of S1
STRIKES      = S1 * STRIKE_PCTS - S2                   # absolute: -10, 0, +10

STRIKE_LABELS = ["90 %", "100 %", "110 %"]
TENOR_LABELS  = ["0.5 Y", "1.0 Y", "1.5 Y"]


# ═══════════════════════════════════════════════════════════════════════════════
#  KIRK'S APPROXIMATION  (1995)
#  Call payoff: max(S1_T – S2_T – K, 0)
#
#  Treats (S2 + K) as a single log-normal "proxy asset".
#  Let w = F2 / (F2 + K),  σ_eff = √(σ1² – 2ρ·σ1·σ2·w + σ2²·w²)
#  Then it reduces to a standard Black call with forward F1/(F2+K), strike 1.
# ═══════════════════════════════════════════════════════════════════════════════
def kirk_price(F1, F2, K, sig1, sig2, rho, T, r=0.0):
    FK = F2 + K
    if FK <= 0:
        # deep ITM: intrinsic value
        return np.exp(-r * T) * max(F1 - F2 - K, 0.0)

    w    = F2 / FK
    s_eff = np.sqrt(max(sig1**2 - 2*rho*sig1*sig2*w + sig2**2*w**2, 1e-14))
    sqrtT = np.sqrt(T)

    F = F1 / FK
    d1 = (np.log(F) + 0.5*s_eff**2*T) / (s_eff*sqrtT)
    d2 = d1 - s_eff*sqrtT

    return np.exp(-r * T) * FK * (F * norm.cdf(d1) - norm.cdf(d2))


# ═══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO PRICER  (single-step geometric Brownian motion)
#  Correlated normals via Cholesky:  Z2 = ρ·Z1 + √(1–ρ²)·Z_indep
# ═══════════════════════════════════════════════════════════════════════════════
def mc_price(S1, S2, K, sig1, sig2, rho, T, r=0.0, n=300_000, seed=None):
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((2, n))
    Z1  = Z[0]
    Z2  = rho * Z[0] + np.sqrt(1.0 - rho**2) * Z[1]

    ST1 = S1 * np.exp((r - 0.5*sig1**2)*T + sig1*np.sqrt(T)*Z1)
    ST2 = S2 * np.exp((r - 0.5*sig2**2)*T + sig2*np.sqrt(T)*Z2)

    payoff = np.maximum(ST1 - ST2 - K, 0.0)
    disc   = np.exp(-r * T)
    price  = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(n)
    return float(price), float(stderr)


# ═══════════════════════════════════════════════════════════════════════════════
#  POPULATE 3 × 3 SURFACE  (tenor × strike)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building spread option surface …")
n_T, n_K = len(TENORS), len(STRIKES)
mc_surf   = np.zeros((n_T, n_K))
mc_err    = np.zeros((n_T, n_K))
kirk_surf = np.zeros((n_T, n_K))

for i, T in enumerate(TENORS):
    for j, K in enumerate(STRIKES):
        F1 = S1 * np.exp(R * T)
        F2 = S2 * np.exp(R * T)
        kirk_surf[i, j]         = kirk_price(F1, F2, K, SIGMA, SIGMA, RHO, T, R)
        mc_surf[i, j], mc_err[i, j] = mc_price(
            S1, S2, K, SIGMA, SIGMA, RHO, T, R,
            n=N_SIMS, seed=42 + i*10 + j
        )
        print(f"  T={T:.1f}Y  K={K:+.0f}  Kirk={kirk_surf[i,j]:.3f}  "
              f"MC={mc_surf[i,j]:.3f} ±{mc_err[i,j]*2:.3f}")

diff_surf = mc_surf - kirk_surf


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0d1117")

fig.suptitle(
    f"Spread Option Surface   ·   S₁ = S₂ = {S1:.0f}   ·   ρ = {RHO:.0%}   ·   "
    f"σ₁ = σ₂ = {SIGMA:.0%}   ·   K = S₁·strike% − S₂",
    color="white", fontsize=13, fontweight="bold", y=0.99
)

DARK_AX  = "#161b22"
GRID_COL = "#30363d"

Tm, Km = np.meshgrid(TENORS, STRIKE_PCTS, indexing="ij")   # (3,3)

# ── helpers ──────────────────────────────────────────────────────────────────
def style_3d(ax, title):
    ax.set_facecolor(DARK_AX)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COL)
    ax.yaxis.pane.set_edgecolor(GRID_COL)
    ax.zaxis.pane.set_edgecolor(GRID_COL)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.set_title(title, color="white", fontweight="bold", pad=8)

def style_2d(ax, title):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.set_title(title, color="white", fontweight="bold")
    ax.spines[["bottom","top","left","right"]].set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.7)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

COLORS = ["#00b4d8", "#f77f00", "#06d6a0"]


# ── 1. Kirk 3-D surface ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0], projection="3d")
surf1 = ax1.plot_surface(Km, Tm, kirk_surf, cmap="plasma",
                         alpha=0.88, edgecolor="k", linewidth=0.3)
ax1.scatter(Km.ravel(), Tm.ravel(), kirk_surf.ravel(), color="white", s=30, zorder=5)
style_3d(ax1, "Kirk's Approximation")
ax1.set_xlabel("Strike %"); ax1.set_ylabel("Tenor (Y)"); ax1.set_zlabel("Price")
ax1.set_xticks(STRIKE_PCTS); ax1.set_xticklabels(["90%","100%","110%"])
ax1.set_yticks(TENORS)
cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.12)
cb1.set_label("Price", color="white")
cb1.ax.yaxis.set_tick_params(color="white", labelcolor="white")


# ── 2. MC 3-D surface ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1], projection="3d")
surf2 = ax2.plot_surface(Km, Tm, mc_surf, cmap="viridis",
                         alpha=0.88, edgecolor="k", linewidth=0.3)
ax2.scatter(Km.ravel(), Tm.ravel(), mc_surf.ravel(), color="white", s=30, zorder=5)
style_3d(ax2, f"Monte Carlo  (n = {N_SIMS:,})")
ax2.set_xlabel("Strike %"); ax2.set_ylabel("Tenor (Y)"); ax2.set_zlabel("Price")
ax2.set_xticks(STRIKE_PCTS); ax2.set_xticklabels(["90%","100%","110%"])
ax2.set_yticks(TENORS)
cb2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.12)
cb2.set_label("Price", color="white")
cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white")


# ── 3. Difference MC – Kirk ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2], projection="3d")
surf3 = ax3.plot_surface(Km, Tm, diff_surf, cmap="RdBu_r",
                         alpha=0.88, edgecolor="k", linewidth=0.3)
ax3.scatter(Km.ravel(), Tm.ravel(), diff_surf.ravel(), color="white", s=30, zorder=5)
style_3d(ax3, "MC − Kirk  (Diff)")
ax3.set_xlabel("Strike %"); ax3.set_ylabel("Tenor (Y)"); ax3.set_zlabel("Δ Price")
ax3.set_xticks(STRIKE_PCTS); ax3.set_xticklabels(["90%","100%","110%"])
ax3.set_yticks(TENORS)
cb3 = fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.12)
cb3.set_label("Δ Price", color="white")
cb3.ax.yaxis.set_tick_params(color="white", labelcolor="white")


# ── 4. Price vs Strike slices (one line per tenor) ──────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_2d(ax4, "Price vs Strike  (per tenor)")
for i, (T, c) in enumerate(zip(TENORS, COLORS)):
    ax4.plot(STRIKE_LABELS, kirk_surf[i], "o--", color=c, lw=2,
             label=f"Kirk {T}Y", alpha=0.95)
    ax4.errorbar(STRIKE_LABELS, mc_surf[i], yerr=2*mc_err[i], fmt="s-",
                 color=c, lw=1.4, capsize=5, label=f"MC {T}Y", alpha=0.75)
ax4.set_xlabel("Strike"); ax4.set_ylabel("Price")
leg = ax4.legend(fontsize=8, ncol=2, facecolor=DARK_AX, edgecolor=GRID_COL,
                 labelcolor="white")


# ── 5. Price vs Tenor slices (one line per strike) ──────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_2d(ax5, "Price vs Tenor  (per strike)")
for j, (lbl, c) in enumerate(zip(STRIKE_LABELS, COLORS)):
    ax5.plot(TENOR_LABELS, kirk_surf[:, j], "o--", color=c, lw=2,
             label=f"Kirk K={lbl}", alpha=0.95)
    ax5.errorbar(TENOR_LABELS, mc_surf[:, j], yerr=2*mc_err[:, j], fmt="s-",
                 color=c, lw=1.4, capsize=5, label=f"MC K={lbl}", alpha=0.75)
ax5.set_xlabel("Tenor"); ax5.set_ylabel("Price")
ax5.legend(fontsize=8, ncol=2, facecolor=DARK_AX, edgecolor=GRID_COL,
           labelcolor="white")


# ── 6. Numeric summary heatmap ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(DARK_AX)
ax6.set_title("MC vs Kirk – Numeric Grid", color="white", fontweight="bold")
ax6.axis("off")

col_hdrs = [""] + STRIKE_LABELS
rows_data = []
for i, T in enumerate(TENORS):
    row = [f"{T}Y"]
    for j in range(n_K):
        rows_data.append(None)          # placeholder; we build text below
        row.append(
            f"MC  {mc_surf[i,j]:.2f}\n"
            f"K   {kirk_surf[i,j]:.2f}\n"
            f"Δ  {diff_surf[i,j]:+.3f}"
        )
    rows_data = rows_data[:-n_K]        # clear placeholder

row_list = []
for i, T in enumerate(TENORS):
    r_ = [f"{T}Y"]
    for j in range(n_K):
        r_.append(
            f"MC  {mc_surf[i,j]:.2f}\n"
            f"K   {kirk_surf[i,j]:.2f}\n"
            f"Δ  {diff_surf[i,j]:+.3f}"
        )
    row_list.append(r_)

tbl = ax6.table(
    cellText=row_list, colLabels=col_hdrs,
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.25, 3.5)

# style table cells
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID_COL)
    if ri == 0:                                      # header row
        cell.set_facecolor("#21262d")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    elif ci == 0:                                    # label column
        cell.set_facecolor("#21262d")
        cell.get_text().set_color(COLORS[ri - 1])
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor(DARK_AX)
        cell.get_text().set_color("white")


# ─── final render ─────────────────────────────────────────────────────────────
out = "spread_option_surface.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out}")
