# MC Calibration — Spread Option Surface

Prices a **spread option** `max(S₁ − S₂ − K, 0)` using two methods — **Kirk's (1995) closed-form approximation** and **Monte Carlo simulation** — then runs a **Differential Evolution optimisation** to calibrate `σ₁`, `σ₂`, and `ρ` back from the Kirk surface.

---

## Overview

| Component | Description |
|---|---|
| **Kirk's Approximation** | Closed-form spread option pricer (Kirk, 1995). Treats `(S₂ + K)` as a single log-normal proxy asset, reducing to a standard Black call. |
| **Monte Carlo Pricer** | Single-step GBM with 300 000 paths. Correlated normals via Cholesky decomposition. |
| **3×3 Surface** | Prices computed across 3 tenors × 3 strikes for both methods. |
| **DE Calibration** | `scipy` Differential Evolution fits `(σ₁, σ₂, ρ)` to match a target Kirk surface. |

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `S1` | 100 | Spot price of asset 1 |
| `S2` | 100 | Spot price of asset 2 |
| `TENORS` | 0.5Y, 1.0Y, 1.5Y | Option maturities |
| `STRIKES` | −10, 0, +10 | Absolute spread strikes (= S1·k% − S2) |
| `sig1` (init) | 30% | Initial guess for σ₁ in calibration |
| `sig2` (init) | 10% | Initial guess for σ₂ in calibration |
| `rho` (init) | 0.5 | Initial guess for correlation ρ |
| `N_SIMS` | 300 000 | Monte Carlo paths |

---

## Files

```
.
├── spread_option_surface.ipynb       # Main notebook (18 cells)
├── spread_option_surface.py          # Standalone Python script
├── spread_option_surface.png         # 3D Kirk vs MC surface plot
├── spread_option_calibration.png     # Calibration fit plot
├── project_log_2026-03-12.txt        # Full session log
└── README.md
```

---

## Notebook Structure

| Section | Content |
|---|---|
| §1 | Imports (`numpy`, `matplotlib`, `scipy`) |
| §2 | Parameters |
| §3 | `kirk_price()` — Kirk (1995) closed-form |
| §4 | `mc_price()` — Monte Carlo pricer |
| §5 | Populate 3×3 surface (Kirk + MC) |
| §6 | 3D surface visualisation |
| §7 | Calibration: DE optimiser recovers `σ₁`, `σ₂`, `ρ` |

---

## Quickstart

```bash
pip install numpy scipy matplotlib
```

**Notebook:**
```
jupyter notebook spread_option_surface.ipynb
# Kernel → Restart & Run All
```

**Standalone script:**
```bash
python spread_option_surface.py
```

---

## Method Notes

### Kirk's Approximation
Let `w = F₂ / (F₂ + K)` and `σ_eff = √(σ₁² − 2ρσ₁σ₂w + σ₂²w²)`.
The spread option price reduces to a standard Black call with forward `F₁/(F₂+K)` and strike 1.

### Monte Carlo
Correlated log-normal paths:
```
Z₂ = ρ·Z₁ + √(1−ρ²)·Z_indep
S_T = S · exp((r − ½σ²)T + σ√T · Z)
payoff = max(S₁_T − S₂_T − K, 0)
```

### Calibration
Differential Evolution minimises the RMSE between `mc_surface(σ₁, σ₂, ρ)` and the Kirk target surface over bounds:
- `σ₁ ∈ [5%, 80%]`
- `σ₂ ∈ [5%, 80%]`
- `ρ ∈ [−0.99, 0.99]`

---

## Output

![Surface Plot](spread_option_surface.png)

![Calibration Plot](spread_option_calibration.png)
