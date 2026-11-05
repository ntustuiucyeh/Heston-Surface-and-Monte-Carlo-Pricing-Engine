"""
Heston Monte Carlo pricer for the autocallable contingent coupon note: date handling, payoff rules,
the simulation, the implied issuer funding spread and the convergence statistics.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# generate all date to year fraction
def year_fraction(
    N: int, pricing_date:pd.Timestamp, maturity_date:pd.Timestamp,
    observation_date:list, autocall_date:list, payment_date:list):
    """
    Convert the product dates to year fractions from the pricing date (ACT/365).

    Parameters
    ----------
    N                : number of time steps
    pricing_date     : pricing date (t = 0)
    maturity_date    : last date of the simulation (the final observation date)
    observation_date : coupon observation dates; list of Timestamps
    autocall_date    : autocall observation dates; list of Timestamps
    payment_date     : payment date of each coupon observation; list of Timestamps

    Returns
    -------
    (T, delta_t, observation_t, autocall_t, payment_t), where T is the horizon in years,
    delta_t = T / N, and the others are arrays of year fractions.
    """

    T = (maturity_date - pricing_date).days / 365

    delta_t = T / N

    observation_t = (
        pd.to_datetime(observation_date) - pricing_date).days / 365

    autocall_t = (
        pd.to_datetime(autocall_date) - pricing_date).days / 365

    payment_t = (
        pd.to_datetime(payment_date) - pricing_date).days / 365


    return T, delta_t, observation_t, autocall_t, payment_t

# observation payoff generator
def obs_payoff_1(
    principal:float, s_i:float, r:float, 
    t_i:float, p_i:float, a_i:float, 
    call_level:float, coupon_level:float, coupon_rate:float):
    """
    Discounted cash flow of Product 1 on one coupon observation date.

    - Autocall date and S >= call level : principal + coupon, and the note terminates
    - otherwise, S >= coupon level      : coupon only
    - otherwise                         : nothing

    Parameters
    ----------
    principal    : principal (notional)
    s_i          : index level on the observation date
    r            : discount rate (risk-free rate, plus the funding spread if any)
    t_i          : observation time (years)
    p_i          : payment time of this observation (years); cash flows are discounted from here
    a_i          : autocall time if t_i is an autocall date, otherwise -1
    call_level   : autocall level (index points)
    coupon_level : coupon barrier (index points)
    coupon_rate  : coupon per observation, as a fraction of principal

    Returns
    -------
    (discounted cash flow, called), where called is True if the note is autocalled.
    """

    if abs(t_i - a_i) < 0.00001 and s_i >= call_level:

        return np.exp(-r*p_i) * (principal + principal * coupon_rate), True
    
    elif s_i >= coupon_level:

        return np.exp(-r*p_i) * (principal * coupon_rate), False

    else:

        return 0, False

# maturity payoff generator
def maturity_payoff_1(
    principal:float, s_0:float, 
    s_i:float, r:float, T:float,
    threshold_level:float):
    """
    Discounted principal repayment of Product 1 at maturity, if the note was not called.

    - S_T >= threshold level : principal
    - S_T <  threshold level : principal * (1 + (S_T - S_0) / S_0) = principal * S_T / S_0

    Parameters
    ----------
    principal       : principal (notional)
    s_0             : initial index level
    s_i             : index level on the final observation date
    r               : discount rate (risk-free rate, plus the funding spread if any)
    T               : payment time of the redemption (maturity date, years)
    threshold_level : downside threshold (index points)

    Returns
    -------
    Discounted redemption amount.
    """

    if s_i >= threshold_level:

        return np.exp(-r*T)*principal

    elif s_i < threshold_level:

        s_return = (s_i - s_0) / s_0

        return np.exp(-r*T)*(principal + (principal * s_return))

# stock process (i+1), q = continuous dividend yield
def s_i_plus_1(s_i:float, r:float, v_i:float, delta_t:float, z_1:float, q:float = 0.0):
    """
    One log-Euler step of the index under Heston, with the variance floored at zero:

        S_{i+1} = S_i * exp[ (r - q - v_i^+ / 2) * dt + sqrt(v_i^+ * dt) * Z_1 ],  v_i^+ = max(v_i, 0)

    Parameters
    ----------
    s_i     : index level at step i
    r       : risk-free rate
    v_i     : variance at step i (may be negative in discrete time)
    delta_t : time step (years)
    z_1     : standard normal draw driving the index
    q       : continuous dividend yield (default 0)

    Returns
    -------
    Index level at step i + 1.
    """

    v_i_plus = max(0, v_i)
    next_s = s_i*np.exp((r - q - 0.5*v_i_plus)*delta_t + np.sqrt(v_i_plus*delta_t)*z_1)

    return next_s

# variance process (i+1)
def v_i_plus_1(v_i:float, kappa:float, theta:float,
               eta:float, rho:float, delta_t:float,
               z_1:float, z_2:float):
    """
    One full-truncation Euler step of the Heston variance:

        v_{i+1} = v_i + kappa * (theta - v_i^+) * dt
                  + eta * sqrt(v_i^+ * dt) * (rho * Z_1 + sqrt(1 - rho^2) * Z_2),  v_i^+ = max(v_i, 0)

    The drift and diffusion use v_i^+, but the update is added to v_i itself (full truncation).
    rho * Z_1 + sqrt(1 - rho^2) * Z_2 gives the variance shock correlation rho with the index shock.

    Parameters
    ----------
    v_i      : variance at step i
    kappa    : speed of mean reversion
    theta    : long-run variance
    eta      : volatility of variance (vol of vol)
    rho      : correlation between the index and its variance
    delta_t  : time step (years)
    z_1, z_2 : independent standard normal draws (z_1 is the same draw used for the index)

    Returns
    -------
    Variance at step i + 1.
    """

    v_i_plus = max(0, v_i)

    next_v = v_i + (kappa*(theta - v_i_plus)*delta_t + 
                    eta*np.sqrt(v_i_plus*delta_t)*(rho*z_1 + np.sqrt(1 - rho**2)*z_2))

    return next_v

# simulate process
def monte_carlo_1(
    M:int, N:int,
    T:float, delta_t:float, observation_t:list, autocall_t:list, payment_t:list,
    s_0:float, v_0:float, r:float, heston_params: list,
    principal:float, call_level:float, coupon_level:float, coupon_rate:float, threshold_level:float,
    q:float = 0.0, funding_spread:float = 0.0):
    """
    Monte Carlo price of Product 1 (autocallable contingent coupon note) under Heston.

    For each of M paths, simulate the index and variance on N steps (s_i_plus_1, v_i_plus_1).
    On each coupon observation date add the discounted coupon or autocall cash flow
    (obs_payoff_1) and stop the path if the note is called. If the note reaches the last
    step, add the discounted redemption (maturity_payoff_1). The price is the average
    path value. Random draws use a fixed seed (42), so results are reproducible.

    Parameters
    ----------
    M, N            : number of paths, number of time steps
    T, delta_t      : horizon (years) and time step, from year_fraction
    observation_t   : coupon observation times (years)
    autocall_t      : autocall observation times (years)
    payment_t       : payment time of each observation (years); the last one is maturity
    s_0, v_0        : initial index level and initial variance
    r               : risk-free rate
    heston_params   : [kappa, theta, eta, rho]
    principal       : principal (notional)
    call_level      : autocall level (index points)
    coupon_level    : coupon barrier (index points)
    coupon_rate     : coupon per observation, as a fraction of principal
    threshold_level : downside threshold at maturity (index points)
    q               : continuous dividend yield; the index drifts at r - q (default 0)
    funding_spread  : issuer funding spread; cash flows are discounted at r + funding_spread (default 0)

    Returns
    -------
    (final_value, product_value, s_sample_path, v_sample_path)
        final_value   : MC price, the mean of product_value
        product_value : discounted value of each path; list of length M
        s_sample_path : index path of each simulation (stops at the autocall date if called)
        v_sample_path : variance path of each simulation
    """

    kappa, theta, eta, rho = heston_params

    # cash flows are discounted at the issuer's funding rate
    r_disc = r + funding_spread

    s_sample_path = []
    v_sample_path = []
    product_value = []

    rng = np.random.default_rng(seed=42)

    Z_1 = rng.standard_normal((M, N))
    Z_2 = rng.standard_normal((M, N))

    for i in range(M):

        value = 0
        call_flag = 0
        obs_idx = 0

        s = [s_0] + []
        v = [v_0] + []

        for j in range(N):

            t_j = (j + 1)*delta_t

            z_1 = Z_1[i, j]
            z_2 = Z_2[i, j]

            s_j = s_i_plus_1(s[j], r, v[j], delta_t, z_1, q)
            v_j = v_i_plus_1(v[j], kappa, theta, eta, rho, 
                             delta_t, z_1, z_2)

            s.append(s_j)
            v.append(v_j)

            # hit observation date
            if obs_idx < len(observation_t) and abs(t_j - observation_t[obs_idx]) < 0.00001:

                # is this observation date also an autocall date?
                a_i = -1.0                                  # -1 means "not an autocall date"
                for a in autocall_t:
                    if abs(t_j - a) < 0.00001:
                        a_i = a

                obs_payoff, call_flag = obs_payoff_1(
                            principal, s_j, r_disc, t_j,
                            payment_t[obs_idx], a_i, 
                            call_level, coupon_level, coupon_rate)

                value = value + obs_payoff

                obs_idx += 1

                # if called
                if call_flag:

                    product_value.append(value)

                    break

            # reach maturity
            if j == N - 1:

                maturity_payoff = maturity_payoff_1(
                    principal, s_0, s_j, r_disc, payment_t[-1], threshold_level)

                value = value + maturity_payoff

                product_value.append(value)

        s_sample_path.append(s)
        v_sample_path.append(v)

    # compute final value
    final_value = np.mean(product_value)

    return final_value, product_value, s_sample_path, v_sample_path

# issuer funding spread s that makes the MC price equal a target (e.g. the issuer's estimated value)
# reuses the simulated paths: only the discount rate r + s changes
def implied_funding_spread(
    target:float, s_sample_path:list, delta_t:float,
    observation_t:list, autocall_t:list, payment_t:list,
    s_0:float, r:float,
    principal:float, call_level:float, coupon_level:float, coupon_rate:float, threshold_level:float):
    """
    Issuer funding spread s that makes the MC price equal a target value, e.g. the issuer's
    estimated value.

    The simulated paths do not depend on the discount rate, so the undiscounted cash flows
    (amount, payment time) of every path are collected once from s_sample_path, and s is
    solved by Brent's method on s in [-5%, 20%]:

        (1 / M) * sum over all cash flows of amount * exp(-(r + s) * payment time) = target

    Parameters
    ----------
    target          : value to match (e.g. 993.00)
    s_sample_path   : index paths returned by monte_carlo_1
    delta_t         : time step used in the simulation (years)
    observation_t   : coupon observation times (years)
    autocall_t      : autocall observation times (years)
    payment_t       : payment time of each observation (years); the last one is maturity
    s_0             : initial index level
    r               : risk-free rate
    principal       : principal (notional)
    call_level      : autocall level (index points)
    coupon_level    : coupon barrier (index points)
    coupon_rate     : coupon per observation, as a fraction of principal
    threshold_level : downside threshold at maturity (index points)

    Returns
    -------
    Implied funding spread s (decimal, e.g. 0.0042 = 0.42%).
    """

    obs_step = [int(round(t / delta_t)) for t in observation_t]

    # undiscounted cash flows (amount, payment time) of every path
    amounts = []
    times = []

    for s in s_sample_path:

        for k, step in enumerate(obs_step):

            is_call_date = any(abs(observation_t[k] - a) < 0.00001 for a in autocall_t)

            if is_call_date and s[step] >= call_level:
                amounts.append(principal * (1 + coupon_rate))
                times.append(payment_t[k])
                break

            if s[step] >= coupon_level:
                amounts.append(principal * coupon_rate)
                times.append(payment_t[k])

            # maturity redemption (note was not called)
            if k == len(obs_step) - 1:
                s_T = s[step]
                amounts.append(principal if s_T >= threshold_level else principal * s_T / s_0)
                times.append(payment_t[-1])

    amounts = np.asarray(amounts)
    times = np.asarray(times)
    M = len(s_sample_path)

    price = lambda spread: (amounts * np.exp(-(r + spread) * times)).sum() / M

    return brentq(lambda spread: price(spread) - target, -0.05, 0.20)

# 99% confidence interval of the MC price (CLT)
def confidence_interval(product_value:list, alpha:float = 0.01):
    """
    Confidence interval of the MC price from the CLT:

        mean +/- z_{1 - alpha/2} * sample std / sqrt(M)

    Parameters
    ----------
    product_value : discounted value of each path
    alpha         : 1 - confidence level (default 0.01, a 99% interval)

    Returns
    -------
    (lower, upper)
    """

    x = np.asarray(product_value)
    M = len(x)

    mean = x.mean()
    half_width = norm.ppf(1 - alpha / 2) * x.std(ddof=1) / np.sqrt(M)

    return mean - half_width, mean + half_width

# WLLN (Chebyshev) upper bound on P(|MC price - true price| >= epsilon)
# P(|mean_M - mu| >= epsilon) <= sigma^2 / (M * epsilon^2)
def chebyshev_bound(product_value:list, epsilon:float):
    """
    Upper bound on the probability that the MC price misses the true price by epsilon or more,
    from the weak law of large numbers (Chebyshev's inequality):

        P(|mean_M - mu| >= epsilon) <= sigma^2 / (M * epsilon^2)

    sigma^2 is estimated by the sample variance; the bound is capped at 1.

    Parameters
    ----------
    product_value : discounted value of each path
    epsilon       : pricing tolerance (same units as the price, e.g. 10 dollars)

    Returns
    -------
    Upper bound on the failure probability.
    """

    x = np.asarray(product_value)
    M = len(x)

    return min(1.0, x.var(ddof=1) / (M * epsilon**2))
