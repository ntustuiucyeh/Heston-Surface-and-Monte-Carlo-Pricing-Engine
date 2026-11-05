"""
Time-step checks for the Heston Monte Carlo: simulations on nested and uniform grids, the note price
against the number of steps, and a European-option check against the Heston closed form.
"""

import numpy as np
import pandas as pd

from modules import calibration

# Heston paths on nested time grids that share the same Brownian motion
# K_list: steps per observation interval, each a power of 2 dividing max(K_list)
# coarse increments are sums of fine increments, so the levels differ only by discretization error
def nested_heston_obs(
    M:int, observation_t:list, K_list:list,
    s_0:float, v_0:float, r:float, heston_params:list, seed:int=42):
    """
    Heston paths on nested time grids that share the same Brownian motion.

    Each observation interval is split into K_max fine steps. A grid with K steps per interval sums blocks
    of K_max / K fine Brownian increments, so every grid is driven by the same Brownian path and their
    prices differ only by discretization error (a paired comparison with little noise).
    Scheme: log-Euler for S, full-truncation Euler for v.

    Parameters
    ----------
    M             : number of paths
    observation_t : observation times (years); the grid restarts at each one
    K_list        : steps per observation interval to compare; each must divide max(K_list)
                    (e.g. powers of 2)
    s_0           : initial index level
    v_0           : initial variance
    r             : drift of the index; pass r - q to include a dividend yield
    heston_params : [kappa, theta, eta, rho]
    seed          : random seed

    Returns
    -------
    Dict {K: array (M, number of observation dates)} of index levels on the observation dates.
    """

    kappa, theta, eta, rho = heston_params
    rho_bar = np.sqrt(1 - rho**2)

    K_list = sorted(K_list)
    K_max = K_list[-1]
    rng = np.random.default_rng(seed=seed)

    s = {K: np.full(M, s_0) for K in K_list}
    v = {K: np.full(M, v_0) for K in K_list}
    s_obs = {K: np.zeros((M, len(observation_t))) for K in K_list}

    t_prev = 0.0
    for k, t_k in enumerate(observation_t):

        dt_fine = (t_k - t_prev) / K_max
        W1 = {K: np.zeros(M) for K in K_list}
        W2 = {K: np.zeros(M) for K in K_list}

        for n in range(1, K_max + 1):

            dW1 = np.sqrt(dt_fine) * rng.standard_normal(M)
            dW2 = np.sqrt(dt_fine) * rng.standard_normal(M)

            for K in K_list:

                W1[K] += dW1
                W2[K] += dW2
                block = K_max // K

                # one step of the K-level scheme (full truncation Euler, log-Euler for S)
                if n % block == 0:
                    dt = block * dt_fine
                    v_plus = np.maximum(0, v[K])
                    s[K] = s[K] * np.exp((r - 0.5*v_plus)*dt + np.sqrt(v_plus)*W1[K])
                    v[K] = v[K] + kappa*(theta - v_plus)*dt + eta*np.sqrt(v_plus)*(rho*W1[K] + rho_bar*W2[K])
                    W1[K][:] = 0
                    W2[K][:] = 0

        for K in K_list:
            s_obs[K][:, k] = s[K]

        t_prev = t_k

    return s_obs

# Heston paths on a uniform grid with N steps (same scheme as monte_carlo_1)
# draws normals one step at a time, so memory does not grow with N
# observation dates are mapped to the nearest step
def heston_obs_paths_N(
    M:int, N:int, T:float, observation_t:list,
    s_0:float, v_0:float, r:float, heston_params:list, seed:int=42):
    """
    Heston paths on a uniform grid with N steps over [0, T], keeping the index on the observation dates.

    Same scheme as monte_carlo.monte_carlo_1 (log-Euler for S, full-truncation Euler for v), but the
    normals are drawn one step at a time, so memory does not grow with N. Observation dates are mapped
    to the nearest step.

    Parameters
    ----------
    M             : number of paths
    N             : number of time steps
    T             : horizon (years)
    observation_t : observation times (years)
    s_0           : initial index level
    v_0           : initial variance
    r             : drift of the index; pass r - q to include a dividend yield
    heston_params : [kappa, theta, eta, rho]
    seed          : random seed

    Returns
    -------
    Array (M, number of observation dates) of index levels on the observation dates.
    """

    kappa, theta, eta, rho = heston_params
    rho_bar = np.sqrt(1 - rho**2)

    delta_t = T / N
    obs_step = {int(round(t / delta_t)): k for k, t in enumerate(observation_t)}

    rng = np.random.default_rng(seed=seed)

    s = np.full(M, s_0)
    v = np.full(M, v_0)
    s_obs = np.zeros((M, len(observation_t)))

    for j in range(1, N + 1):

        z_1 = rng.standard_normal(M)
        z_2 = rng.standard_normal(M)

        v_plus = np.maximum(0, v)
        s = s * np.exp((r - 0.5*v_plus)*delta_t + np.sqrt(v_plus*delta_t)*z_1)
        v = v + kappa*(theta - v_plus)*delta_t + eta*np.sqrt(v_plus*delta_t)*(rho*z_1 + rho_bar*z_2)

        if j in obs_step:
            s_obs[:, obs_step[j]] = s

    return s_obs

# product price for each N on a uniform grid, with 99% CI
def price_by_N(M:int, N_list:list, T:float, observation_t:list,
               s_0:float, v_0:float, r:float, heston_params:list, value_fn, z:float=2.5758):
    """
    Note price for each number of time steps N on a uniform grid, with the 99% CI half-width.

    Each N is simulated separately with heston_obs_paths_N (same seed), and value_fn turns the observed
    index levels into discounted path payoffs.

    Parameters
    ----------
    M             : number of paths
    N_list        : numbers of time steps to try
    T             : horizon (years)
    observation_t : observation times (years)
    s_0           : initial index level
    v_0           : initial variance
    r             : drift of the index; pass r - q to include a dividend yield
    heston_params : [kappa, theta, eta, rho]
    value_fn      : function mapping s_obs (M, number of observation dates) to M discounted path payoffs
    z             : normal quantile for the CI (2.5758 = 99%)

    Returns
    -------
    DataFrame with columns 'N', 'price' and '99% half-width'.
    """

    rows = []

    for N in N_list:

        x = value_fn(heston_obs_paths_N(M, N, T, observation_t, s_0, v_0, r, heston_params))
        rows.append({'N': N, 'price': x.mean(), '99% half-width': z * x.std(ddof=1) / np.sqrt(M)})

    return pd.DataFrame(rows)

# product price for each grid level, with 99% CI and the paired difference to the finest grid
def price_vs_N(s_obs:dict, value_fn, n_intervals:int, z:float=2.5758):
    """
    Note price on each nested grid, with its CI and the paired difference to the finest grid.

    The grids from nested_heston_obs share the same Brownian motion, so the difference to the finest
    grid isolates the discretization error, with a much narrower CI than the prices themselves.

    Parameters
    ----------
    s_obs       : {K: array (M, number of observation dates)} from nested_heston_obs
    value_fn    : function mapping s_obs (M, number of observation dates) to M discounted path payoffs
    n_intervals : number of observation intervals, so that N = n_intervals * K
    z           : normal quantile for the CI (2.5758 = 99%)

    Returns
    -------
    DataFrame with columns 'steps per interval', 'N', 'price', '99% half-width', 'diff to finest'
    and 'diff 99% half-width'.
    """

    values = {K: value_fn(paths) for K, paths in s_obs.items()}
    finest = values[max(values)]

    rows = []

    for K, x in values.items():

        diff = x - finest
        rows.append({'steps per interval': K,
                     'N': n_intervals * K,
                     'price': x.mean(),
                     '99% half-width': z * x.std(ddof=1) / np.sqrt(len(x)),
                     'diff to finest': diff.mean(),
                     'diff 99% half-width': z * diff.std(ddof=1) / np.sqrt(len(x))})

    return pd.DataFrame(rows)

# Heston closed-form European put/call, discounted (theoretical benchmark)
def heston_european(s_0:float, K:float, T:float, r:float, v_0:float, heston_params:list, option:str='put'):
    """
    Discounted Heston price of a European call or put from the semi-closed form.

    Uses calibration.heston_price with forward F = s_0 * exp(r T) and a fine integration grid
    (20,000 points), then discounts at r. Used as the exact benchmark in european_check.

    Parameters
    ----------
    s_0           : initial index level
    K             : strike
    T             : maturity (years)
    r             : risk-free rate (no dividend)
    v_0           : initial variance
    heston_params : [kappa, theta, eta, rho]
    option        : 'call' or 'put'

    Returns
    -------
    Option price today.
    """

    kappa, theta, eta, rho = heston_params
    F = s_0 * np.exp(r * T)

    return np.exp(-r * T) * calibration.heston_price(F, K, T, v_0, kappa, theta, eta, rho, option, u_max=400.0, N=20000)

# MC European prices at maturity for each grid level, against the closed form
def european_check(s_obs:dict, s_0:float, T:float, r:float, v_0:float, heston_params:list, strikes:dict):
    """
    Monte Carlo European prices for each grid level, against the Heston closed form.

    For each grid, prices the given options from the simulated S_T and reports the MC price, the exact
    price, the error and the 99% CI half-width. Also checks the martingale property
    E[exp(-rT) S_T] = S_0. A good scheme has errors inside the CI.

    Parameters
    ----------
    s_obs         : {K: array (M, number of observation dates)} from nested_heston_obs; the last column
                    is used as S_T
    s_0           : initial index level
    T             : time to the last observation date (years)
    r             : risk-free rate (no dividend), used for the drift and for discounting
    v_0           : initial variance
    heston_params : [kappa, theta, eta, rho]
    strikes       : {label: (strike, 'call' or 'put')}

    Returns
    -------
    DataFrame with one row per grid level: '<label> MC', '<label> exact', '<label> error' and
    '<label> 99% half-width' for each option, and 'E[disc S_T] - S0'.
    """

    rows = []

    for K_steps, paths in s_obs.items():

        s_T = paths[:, -1]
        row = {'steps per interval': K_steps}

        for label, (strike, option) in strikes.items():

            payoff = np.maximum(strike - s_T, 0) if option == 'put' else np.maximum(s_T - strike, 0)
            x = np.exp(-r * T) * payoff
            exact = heston_european(s_0, strike, T, r, v_0, heston_params, option)
            row[f'{label} MC'] = x.mean()
            row[f'{label} exact'] = exact
            row[f'{label} error'] = x.mean() - exact
            row[f'{label} 99% half-width'] = 2.5758 * x.std(ddof=1) / np.sqrt(len(x))

        # martingale check: E[e^{-rT} S_T] = S_0
        row['E[disc S_T] - S0'] = np.exp(-r * T) * s_T.mean() - s_0
        rows.append(row)

    return pd.DataFrame(rows)
