"""
Cox-Ross-Rubinstein binomial tree for the autocallable contingent coupon note under a flat volatility,
used as an independent check of the Monte Carlo pricer, plus the flat-volatility lookup on the OVDV surface.
"""

import numpy as np
import pandas as pd

# map year fractions to the nearest tree step
def time_to_step(t_list, delta:float):
    """
    Map year fractions to the nearest step of a grid with step size delta.

    Parameters
    ----------
    t_list : times in years; list or array
    delta  : step size (years)

    Returns
    -------
    List of integer step indices, round(t / delta) for each t.
    """

    return [int(round(t / delta)) for t in t_list]

# CRR binomial tree for the autocallable contingent coupon note
# payoff convention follows monte_carlo.monte_carlo_1
def crr_autocall(
    n:int, T:float, observation_t:list, autocall_t:list, payment_t:list,
    s_0:float, r:float, q:float, sigma:float,
    principal:float, call_level:float, coupon_level:float, coupon_rate:float, threshold_level:float):
    """
    Price the autocallable contingent coupon note on a CRR binomial tree with a flat volatility.

    Backward induction from the final observation date. On each observation date a coupon is added where
    S >= coupon level, and on autocall dates the node value is replaced by principal + coupon where
    S >= call level. Each cash flow is discounted from its payment date back to its observation date, so
    the payoff matches monte_carlo.monte_carlo_1.

        u = exp(sigma * sqrt(delta)),  d = 1 / u,  q_u = (exp((r - q) * delta) - d) / (u - d)

    Parameters
    ----------
    n               : number of tree steps
    T               : time to the final observation date (years)
    observation_t   : coupon observation times (years)
    autocall_t      : autocall observation times (years)
    payment_t       : payment time of each observation (years)
    s_0             : initial index level
    r               : risk-free rate (continuous), used for the drift and for discounting
    q               : continuous dividend yield
    sigma           : flat volatility
    principal       : principal (notional)
    call_level      : autocall level (index level)
    coupon_level    : coupon barrier (index level)
    coupon_rate     : coupon per observation as a fraction of principal
    threshold_level : downside threshold at maturity (index level)

    Returns
    -------
    Present value of the note at t = 0.
    """

    delta = T / n
    u = np.exp(sigma * np.sqrt(delta))
    d = 1 / u
    qu = (np.exp((r - q) * delta) - d) / (u - d)
    qd = 1 - qu
    disc = np.exp(-r * delta)

    coupon = principal * coupon_rate

    obs_step = time_to_step(observation_t, delta)
    call_step = set(time_to_step(autocall_t, delta))

    # step -> payment time of that observation
    obs_pay = dict(zip(obs_step, payment_t))

    # terminal nodes (last observation = valuation date)
    i = np.arange(n + 1)
    s = s_0 * u**i * d**(n - i)

    pay_disc = np.exp(-r * (payment_t[-1] - T))
    redemption = np.where(s >= threshold_level, principal, principal * s / s_0)
    value = pay_disc * (redemption + np.where(s >= coupon_level, coupon, 0.0))

    # backward induction
    for j in range(n - 1, -1, -1):

        value = disc * (qu * value[1:] + qd * value[:-1])

        if j in obs_pay:

            i = np.arange(j + 1)
            s = s_0 * u**i * d**(j - i)
            pay_disc = np.exp(-r * (obs_pay[j] - j * delta))

            value = value + np.where(s >= coupon_level, coupon * pay_disc, 0.0)

            if j in call_step:

                value = np.where(s >= call_level, (principal + coupon) * pay_disc, value)

    return value[0]

# convergence: price with N, 2N, ..., n1max*N steps
def crr_convergence(N:int, n1max:int, **kwargs):
    """
    Price the note on trees with N, 2N, ..., n1max * N steps, to check convergence.

    Parameters
    ----------
    N        : base number of steps (e.g. one per calendar day)
    n1max    : largest multiple of N
    **kwargs : all other arguments of crr_autocall (T, dates, s_0, r, q, sigma and payoff terms)

    Returns
    -------
    DataFrame with columns 'num_steps' and 'Value'.
    """

    result = []

    for n1 in range(1, n1max + 1):

        result.append({'num_steps': n1 * N, 'Value': crr_autocall(n1 * N, **kwargs)})

    return pd.DataFrame(result)

# sensitivity: price across a list of flat volatilities
def crr_vols(n:int, vols:list, **kwargs):
    """
    Price the note on an n-step tree for each flat volatility in vols.

    Parameters
    ----------
    n        : number of tree steps
    vols     : flat volatilities to try; list
    **kwargs : all other arguments of crr_autocall except sigma

    Returns
    -------
    DataFrame with columns 'vol' and 'Value'.
    """

    result = []

    for sigma in vols:

        result.append({'vol': sigma, 'Value': crr_autocall(n, sigma=sigma, **kwargs)})

    return pd.DataFrame(result)

# implied vol from the OVDV surface at maturity T and moneyness K/S
# total variance interpolated linearly in T, iv linear in moneyness (flat-slope extrapolation)
def surface_vol(surface:pd.DataFrame, T:float, moneyness:float):
    """
    Implied volatility from the OVDV surface at maturity T and spot moneyness K / S.

    Within an expiry, IV is linear in moneyness: below the lowest quoted moneyness it is extrapolated with
    the slope of the first two points, and above the highest it is held flat. Across expiries, total
    variance sigma^2 * T is interpolated linearly in T between the two expiries around T (extrapolated
    from the first or last pair outside the grid).

    Parameters
    ----------
    surface   : one row per expiry, with a 'T' column (years, ascending) and IV columns named
                'iv_<moneyness in %>' (e.g. 'iv_80', 'iv_97_5')
    T         : maturity (years)
    moneyness : K / S (e.g. 0.7)

    Returns
    -------
    Implied volatility at (T, moneyness).
    """

    cols = [c for c in surface.columns if c.startswith('iv_')]
    m_grid = np.array([float(c[3:].replace('_', '.')) / 100 for c in cols])

    def iv_row(row):
        """IV of one expiry row at the requested moneyness (linear, slope-extrapolated below the grid)."""
        iv = row[cols].to_numpy(dtype=float)
        if moneyness < m_grid[0]:
            slope = (iv[1] - iv[0]) / (m_grid[1] - m_grid[0])
            return iv[0] + slope * (moneyness - m_grid[0])
        return np.interp(moneyness, m_grid, iv)

    t_grid = surface['T'].to_numpy(dtype=float)
    k = min(max(np.searchsorted(t_grid, T), 1), len(t_grid) - 1)
    t0, t1 = t_grid[k - 1], t_grid[k]
    w0, w1 = iv_row(surface.iloc[k - 1])**2 * t0, iv_row(surface.iloc[k])**2 * t1
    w = w0 + (w1 - w0) * (T - t0) / (t1 - t0)

    return np.sqrt(w / T)
