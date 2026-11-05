# Pricing an S&P 500 Autocallable Note with a Calibrated Heston Model

I price a Bank of America four-year autocallable contingent coupon note on the S&P 500 with a Heston
stochastic volatility model calibrated to the SPX implied volatility surface (Bloomberg), using Monte Carlo
simulation. I then test whether stochastic volatility changes the answer compared with a standard
flat-volatility model.

**Full report:** [report/Heston_Autocallable_Pricing_Report.pdf](report/Heston_Autocallable_Pricing_Report.pdf)

## Key results

- **Fair value: $998.56 per $1,000** (99% CI $995.65 to $1,001.47), 0.56% above BofA's estimated value of
  $993.00. The gap matches an issuer funding spread of **0.40%** over the SOFR OIS rate.
- **The numbers are robust:** with 20,000 paths the probability of a pricing error above $10 is at most 1.3%,
  the price is stable to the time step, and the flat-volatility Monte Carlo agrees with an independent
  binomial tree within $0.76.
- **Flat volatility gets the price but not the risk.** At the ATM volatility, a flat-vol model prices the note
  within $2 of Heston, but the outcomes behind the price are very different:

| | Heston MC | Flat-vol MC (ATM 20.89%) |
|---|---:|---:|
| Fair value | $998.56 | $996.67 |
| Called on the first call date | 67.9% | 52.4% |
| Expected life | 1.60 years | 1.88 years |
| Probability of losing principal | 7.4% | 11.1% |
| Average loss, given a loss | 55% of principal | 44% of principal |

Choosing the volatility at the 70% barrier (25.40%) instead underprices the note by about $31.
All figures are in the [report](report/Heston_Autocallable_Pricing_Report.pdf) and in the notebook outputs.

## Method

1. **Market data** (Bloomberg, 12-Oct-2026 surface; rates and dividends on the 18-Sep-2026 pricing date):
   SOFR OIS rate 4.567%, OVME dividend yield 1.151%, SPX option chains and the ATM volatility term structure.
   Quotes with no bid or fewer than 10 contracts traded are removed; implied volatilities are backed out with
   Black's formula using Bloomberg forwards and discount factors from put-call parity.
2. **Calibration in three steps:** kappa and theta from the ATM variance term structure, eta and rho from the
   Dec-2026 and Mar-2027 smiles, then all five parameters on 200 options across 8 expiries
   (RMSE 0.19 vol points). Spread-weighted and Feller-penalized objectives are compared.
3. **Pricing:** Monte Carlo with 20,000 paths and daily steps (log-Euler for the index, full-truncation Euler
   for the variance), discounting each cash flow at the OIS rate.
4. **Validation:** Chebyshev error bound, 300 independent runs to check the CLT interval, a time-step check,
   and a Cox-Ross-Rubinstein binomial tree as an independent benchmark.
5. **Model comparison:** value decomposition, autocall and loss probabilities, and the loss distribution under
   Heston and flat volatility.

## Repository structure

```
├── data_preprocess.ipynb   # 1. clean quotes, discount factors, implied volatilities
├── calibration.ipynb       # 2. three-step Heston calibration and fit diagnostics
├── monte_carlo.ipynb       # 3. Heston MC price, funding spread, convergence, flat-vol benchmarks
├── comparison.ipynb        # 4. Heston vs flat volatility: decomposition, probabilities, losses
├── modules/
│   ├── preprocess.py       # data reading, cleaning, Black implied volatility
│   ├── calibration.py      # Heston pricing (characteristic function) and calibration
│   ├── monte_carlo.py      # Heston Monte Carlo pricer and convergence statistics
│   ├── binomial.py         # CRR binomial tree for the note, flat-vol surface lookup
│   ├── comparison.py       # vectorized paths, per-path cash flows, tree event probabilities
│   └── discretization.py   # time-step checks
└── report/                 # final report (PDF)
```

## How to run

```bash
pip install -r requirements.txt
```

Put the data files in a `Data/` folder (see below), then run the notebooks in order 1 to 4.
Tested with Python 3.14.

## Data

The Bloomberg market data is **not included** in this repository because Bloomberg data may not be
redistributed. The notebooks expect these files in `Data/`:

| File | Content | Bloomberg source |
|---|---|---|
| `ATM_IV.xlsx` | sheet `ATMF`: ATM-forward IV term structure; sheet `OVDV`: IV by moneyness and expiry | OVDV |
| `Smile_Dec_2026.csv`, `Surface.csv` | raw SPX option chains (bid, ask, volume) | OMON |
| `Calibration_Surface.csv` | 200-option calibration set (8 expiries, 25 strikes each) | OMON |

`data_preprocess.ipynb` creates `IV_Smile_Dec_2026.csv` and `IV_Smile_Mar_2027.csv` from the raw chains.
The rate (USD SOFR OIS, curve S490) and the dividend yield (OVME) are entered directly in the notebooks.

## Limitations

The surface is from 12-Oct-2026 rather than the pricing date, the rate and dividend yield are held constant,
the Feller penalty weight was not tuned, and no Greeks are computed. See Section 7 of the report.
