"""
Tools for comparing the Heston Monte Carlo with flat-volatility models: vectorized Heston paths,
per-path cash flows and event indicators, event probabilities on the binomial tree, and the implied
flat volatility.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from modules import binomial

# vectorized version of monte_carlo.monte_carlo_1 (same scheme, same seed and draws)
def heston_obs_paths(
    M:int, N:int, delta_t:float, observation_t:list,
    s_0:float, v_0:float, r:float, heston_params:list, seed:int=42):
    """
    Simulate Heston index paths and keep the index level on each observation date.

    Vectorized version of monte_carlo.monte_carlo_1: same scheme (log-Euler for S, full-truncation Euler
    for v), same seed and same random draws, so it reproduces the same paths. All M x N normals are drawn
    up front, so memory grows with M * N.

    Setting eta = 0 and v_0 = theta = sigma^2 gives constant-volatility GBM (the flat-vol Monte Carlo).

    Parameters
    ----------
    M             : number of paths
    N             : number of time steps
    delta_t       : step size (years)
    observation_t : observation times (years); each is mapped to the nearest step
    s_0           : initial index level
    v_0           : initial variance
    r             : drift of the index; pass r - q to include a dividend yield
    heston_params : [kappa, theta, eta, rho]
    seed          : random seed

    Returns
    -------
    Array of shape (M, number of observation dates) with the index level on each date.
    """

    kappa, theta, eta, rho = heston_params

    rng = np.random.default_rng(seed=seed)

    Z_1 = rng.standard_normal((M, N))
    Z_2 = rng.standard_normal((M, N))

    obs_step = {int(round(t / delta_t)): k for k, t in enumerate(observation_t)}

    s = np.full(M, s_0)
    v = np.full(M, v_0)
    s_obs = np.zeros((M, len(observation_t)))

    for j in range(N):

        z_1 = Z_1[:, j]
        z_2 = Z_2[:, j]

        v_plus = np.maximum(0, v)
        s = s * np.exp((r - 0.5*v_plus)*delta_t + np.sqrt(v_plus*delta_t)*z_1)
        v = v + (kappa*(theta - v_plus)*delta_t +
                 eta*np.sqrt(v_plus*delta_t)*(rho*z_1 + np.sqrt(1 - rho**2)*z_2))

        if j + 1 in obs_step:
            s_obs[:, obs_step[j + 1]] = s

    return s_obs

# cash flows of each path, split by type, plus event indicators
def mc_cashflows(
    s_obs:np.ndarray, s_0:float, r:float, observation_t:list, autocall_t:list, payment_t:list,
    principal:float, call_level:float, coupon_level:float, coupon_rate:float, threshold_level:float):
    """
    Discounted cash flows of each path, split by type, plus event indicators.

    Applies the note's rules on each observation date (coupon if S >= coupon level; on autocall dates,
    principal + coupon if S >= call level, after which the path stops paying) and the maturity payoff for
    paths never called (principal if S_T >= threshold, otherwise principal * S_T / S_0).

    Parameters
    ----------
    s_obs           : index level on each observation date; array (M, K) from heston_obs_paths
    s_0             : initial index level
    r               : discount rate (risk-free rate, plus the funding spread if any)
    observation_t   : observation times (years)
    autocall_t      : autocall times (years)
    payment_t       : payment time of each observation (years)
    principal       : principal (notional)
    call_level      : autocall level (index level)
    coupon_level    : coupon barrier (index level)
    coupon_rate     : coupon per observation as a fraction of principal
    threshold_level : downside threshold at maturity (index level)

    Returns
    -------
    Dict of per-path arrays:
        'pv_coupon'   : PV of the coupons
        'pv_call'     : PV of the principal repaid at an autocall
        'pv_maturity' : PV of the principal repaid at maturity
        'value'       : total PV (sum of the three)
        'call_date'   : index of the observation where the note is called, -1 if never called
        'coupon_paid' : boolean array (M, K), coupon paid on each observation
        'loss'        : True if never called and S_T < threshold (principal loss)
        's_T'         : index level on the final observation date
    """

    M, K = s_obs.shape
    coupon = principal * coupon_rate
    disc = np.exp(-r * np.asarray(payment_t))

    alive = np.ones(M, dtype=bool)
    pv_coupon = np.zeros(M)
    pv_call = np.zeros(M)
    pv_maturity = np.zeros(M)
    call_date = np.full(M, -1)                  # observation index where called, -1 = not called
    coupon_paid = np.zeros((M, K), dtype=bool)

    for k in range(K):

        s = s_obs[:, k]
        is_call_date = any(abs(observation_t[k] - a) < 0.00001 for a in autocall_t)

        paid = alive & (s >= coupon_level)
        if is_call_date:
            paid |= alive & (s >= call_level)
        coupon_paid[:, k] = paid
        pv_coupon += paid * coupon * disc[k]

        if is_call_date:
            called = alive & (s >= call_level)
            pv_call += called * principal * disc[k]
            call_date[called] = k
            alive &= ~called

    s_T = s_obs[:, -1]
    redemption = np.where(s_T >= threshold_level, principal, principal * s_T / s_0)
    pv_maturity = alive * redemption * disc[-1]
    loss = alive & (s_T < threshold_level)

    return {'pv_coupon': pv_coupon, 'pv_call': pv_call, 'pv_maturity': pv_maturity,
            'value': pv_coupon + pv_call + pv_maturity,
            'call_date': call_date, 'coupon_paid': coupon_paid, 'loss': loss, 's_T': s_T}

# forward pass through the CRR tree: event probabilities and PV split by cash flow type
def crr_event_probs(
    n:int, T:float, observation_t:list, autocall_t:list, payment_t:list,
    s_0:float, r:float, q:float, sigma:float,
    principal:float, call_level:float, coupon_level:float, coupon_rate:float, threshold_level:float):
    """
    Forward pass through the CRR tree: event probabilities and PV split by cash-flow type.

    Propagates the risk-neutral node probabilities forward. On autocall dates the probability of the
    called nodes is removed, so later probabilities are for notes still alive. Uses the same tree and
    payoff rules as binomial.crr_autocall, so 'value' matches its price.

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
    Dict with
        'p_call'      : probability of being called on each observation date (0 where no autocall)
        'p_coupon'    : probability of a coupon on each observation date
        'p_maturity'  : probability of reaching maturity (never called)
        'p_loss'      : probability of a principal loss at maturity
        'pv_coupon'   : PV of the coupons
        'pv_call'     : PV of the principal repaid at an autocall
        'pv_maturity' : PV of the principal repaid at maturity
        'value'       : total PV
    """

    delta = T / n
    u = np.exp(sigma * np.sqrt(delta))
    d = 1 / u
    qu = (np.exp((r - q) * delta) - d) / (u - d)
    qd = 1 - qu

    coupon = principal * coupon_rate
    obs_step = binomial.time_to_step(observation_t, delta)
    call_step = set(binomial.time_to_step(autocall_t, delta))
    disc = np.exp(-r * np.asarray(payment_t))

    p_call = np.zeros(len(observation_t))
    p_coupon = np.zeros(len(observation_t))
    pv_coupon = pv_call = 0.0

    prob = np.array([1.0])
    for j in range(1, n + 1):

        new = np.zeros(j + 1)
        new[1:] += prob * qu
        new[:-1] += prob * qd
        prob = new

        if j in obs_step:

            k = obs_step.index(j)
            i = np.arange(j + 1)
            s = s_0 * u**i * d**(j - i)

            paid = s >= coupon_level
            if j in call_step:
                paid |= s >= call_level
            p_coupon[k] = prob[paid].sum()
            pv_coupon += p_coupon[k] * coupon * disc[k]

            if j in call_step:
                called = s >= call_level
                p_call[k] = prob[called].sum()
                pv_call += p_call[k] * principal * disc[k]
                prob = np.where(called, 0.0, prob)

    i = np.arange(n + 1)
    s = s_0 * u**i * d**(n - i)
    redemption = np.where(s >= threshold_level, principal, principal * s / s_0)
    pv_maturity = (prob * redemption).sum() * disc[-1]

    return {'p_call': p_call, 'p_coupon': p_coupon,
            'p_maturity': prob.sum(), 'p_loss': prob[s < threshold_level].sum(),
            'pv_coupon': pv_coupon, 'pv_call': pv_call, 'pv_maturity': pv_maturity,
            'value': pv_coupon + pv_call + pv_maturity}

# flat vol that makes the binomial price equal a target price (e.g. the Heston MC price)
def implied_flat_vol(target:float, n:int, lo:float=0.05, hi:float=0.60, **kwargs):
    """
    Flat volatility at which the binomial tree price equals a target price (e.g. the Heston MC price).

    Solves binomial.crr_autocall(n, sigma=sigma) = target for sigma by Brent's method on [lo, hi].

    Parameters
    ----------
    target   : target price
    n        : number of tree steps
    lo, hi   : search bracket for sigma
    **kwargs : all other arguments of binomial.crr_autocall except sigma

    Returns
    -------
    Implied flat volatility.
    """

    return brentq(lambda sigma: binomial.crr_autocall(n, sigma=sigma, **kwargs) - target, lo, hi, xtol=1e-6)
