"""
Heston calibration to the SPX implied volatility surface.

Step 1 fits kappa and theta to the ATM variance term structure, Step 2 fits eta and rho to the
short-dated smiles, and Step 3 fits all five parameters to the full surface. Each fit has a regular,
a bid-ask-spread-weighted and a Feller-penalized objective. Also contains the Heston and Black pricing
formulas and the fit-error diagnostics.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq
from scipy.stats import norm

# step 1: estimate v_0, theta, kappa
def heston_ept_avar(T:float, kappa:float, theta: float, v_0: float):
    """
    Heston expected average variance from today to maturity T.

    Under Heston, E[v_t] = theta + (v_0 - theta) * exp(-kappa * t). Averaging over [0, T]:

        avg_var(T) = theta + (v_0 - theta) * (1 - exp(-kappa * T)) / (kappa * T)

    This is compared with the market ATM implied variance sigma_ATM(T)^2.

    Parameters
    ----------
    T     : time to maturity (years); float or array
    kappa : speed of mean reversion
    theta : long-run variance
    v_0   : initial variance

    Returns
    -------
    Expected average variance to T, same shape as T.
    """
    
    f = theta + (v_0 - theta) * ((1 - np.exp(-kappa*T))/ (kappa*T))

    return f

def res_1(params:np.ndarray, T:float, v_0: float, mrk_var:float) -> float:
    """
    Residuals for the step-1 fit: model expected average variance minus market ATM variance.

    Parameters
    ----------
    params  : [kappa, theta], the parameters being fitted
    T       : maturities (years); array
    v_0     : initial variance, held fixed
    mrk_var : market ATM implied variance at each maturity; array

    Returns
    -------
    Array of residuals avg_var(T_i) - mrk_var_i, one per maturity.
    """

    kappa, theta = params

    res = heston_ept_avar(T, kappa, theta, v_0) - mrk_var

    return res

def fit_kappa_theta(
        T:np.ndarray, v_0: float,
        mrk_var:np.ndarray, kappa_0:float, theta_0:float) -> tuple[float, float]:
    """
    Step 1 of the calibration: fit kappa and theta to the ATM variance term structure, with v_0 fixed.

    Solves
        kappa*, theta* = argmin sum_i [ avg_var(T_i; kappa, theta, v_0) - mrk_var_i ]^2
    by bounded least squares, kappa in [0.1, 5] and theta in [0.0001, 1].

    Parameters
    ----------
    T       : maturities (years); array
    v_0     : initial variance, held fixed (e.g. ATM variance of the nearest liquid expiry)
    mrk_var : market ATM implied variance at each maturity; array
    kappa_0 : initial guess for kappa
    theta_0 : initial guess for theta (e.g. ATM variance of the longest maturity)

    Returns
    -------
    (kappa_star, theta_star)
    """

    result = least_squares(
        fun=res_1, x0=[kappa_0, theta_0], args=(T, v_0, mrk_var),
        bounds=([0.1, 0.0001], [5.0, 1.0]))

    kappa_star, theta_star = result.x

    return kappa_star, theta_star 

# step 2: estimate rho, eta

# Heston price
def heston_price(F, K, tau, v0, kappa, theta, eta, rho, option='call',
                 u_max=400.0, N=4000, eps=1e-7):
    """
    Undiscounted (forward) European call or put price under the Heston model.

    Call_fwd = F * P1 - K * P0
    Put_fwd  = Call_fwd - (F - K)        (put-call parity, undiscounted)

    Multiply by the discount factor to get today's price. For calibration
    in implied-vol space this is not needed: Black's formula with df = 1
    gives the same implied volatility.

    Parameters
    ----------
    F, K, tau    : forward price, strike, time to maturity (years)
    v0           : initial variance
    kappa        : speed of mean reversion
    theta        : long-run variance
    eta          : volatility of variance (vol of vol)
    rho          : correlation between the stock and its variance
    option       : 'call' or 'put'
    u_max, N, eps: integration upper limit, number of grid points, lower limit
    """

    # Step 1: log-moneyness
    x = np.log(F / K)

    # Step 2: integration grid
    u = np.linspace(eps, u_max, N)

    # Step 3: compute P0 (j = 0) and P1 (j = 1)
    probabilities = []

    for j in [0, 1]:

        # coefficients of the Riccati equations
        alpha = -0.5 * (u * u - 1j * (2 * j - 1) * u)
        beta = kappa - j * rho * eta - 1j * rho * eta * u

        # d: choose the square root with positive real part
        d = np.sqrt(beta * beta - 2 * eta**2 * alpha)
        d = np.where(np.real(d) < 0, -d, d)

        # g: if |g| >= 1, use the other root so that |g| < 1 (avoids blow-ups)
        g = (beta - d) / (beta + d)
        flip = np.abs(g) >= 1
        d = np.where(flip, -d, d)
        g = np.where(flip, 1 / g, g)

        # C and D of the characteristic function
        exp_term = np.exp(-d * tau)
        D = (beta - d) / eta**2 * (1 - exp_term) / (1 - g * exp_term)
        C = (kappa * theta / eta**2) * ((beta - d) * tau
                                        - 2 * (np.log(1 - g * exp_term) - np.log(1 - g)))

        # characteristic function and the integrand
        phi = np.exp(C + D * v0 + 1j * u * x)
        integrand = np.real(phi / (1j * u))

        # probability P_j
        P_j = 0.5 + np.trapezoid(integrand, u) / np.pi
        probabilities.append(P_j)

    P0 = probabilities[0]
    P1 = probabilities[1]

    # Step 4: undiscounted call price
    call_price = F * P1 - K * P0

    # Step 5: return the call, or the put from put-call parity
    if option == 'call':
        return call_price
    else:
        put_price = call_price - (F - K)
        return put_price

# Black-Scholes price
def black_price(F, K, tau, sigma, option='call'):
    """
    Undiscounted (forward) Black-Scholes price of a European option.

    Call_fwd = F * N(d1) - K * N(d2)
    Put_fwd  = K * N(-d2) - F * N(-d1)

    d1 = [ ln(F/K) + 0.5 * sigma^2 * tau ] / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    Parameters
    ----------
    F, K, tau : forward price, strike, time to maturity (years)
    sigma     : volatility
    option    : 'call' or 'put'
    """

    # Step 1: d1 and d2
    sigma_sqrt_tau = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau

    # Step 2: undiscounted price
    if option == 'call':
        price = F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        price = K * norm.cdf(-d2) - F * norm.cdf(-d1)

    return price

# Inverse Black-Scholes
def black_implied_vol(price, F, K, tau, option='call'):
    """
    Implied volatility from an undiscounted (forward) option price.
    Finds sigma such that black_price(F, K, tau, sigma, option) = price, by Brent's method
    on sigma in [0.0001, 5]. Raises ValueError if no root exists in that range, e.g. if the
    price is below intrinsic value or numerically zero.

    Parameters
    ----------
    price     : undiscounted option price
    F, K, tau : forward price, strike, time to maturity (years)
    option    : 'call' or 'put'

    Returns
    -------
    Black implied volatility.
    """

    # difference between Black price and the target price
    def price_error(sigma):
        return black_price(F, K, tau, sigma, option) - price

    # search for sigma between 0.01% and 500%
    iv = brentq(price_error, 1e-4, 5.0)

    return iv

# regular residual
def res_2(
    params:list, F:np.array, K:np.array, tau:np.array, 
    v0:float, kappa:float, theta:float, option:np.ndarray, mrk_iv:np.ndarray) -> np.ndarray:
    """
    Residuals for the step-2 regular fit: Heston implied vol minus market implied vol.

    For each option i, price it under Heston with (v0, kappa, theta) fixed and the trial (eta, rho),
    back out its Black implied vol, and return

        res_i = sigma_imp_i(Heston) - sigma_imp_i(market)

    Parameters
    ----------
    params       : [eta, rho], the parameters being fitted
    F, K, tau    : forward, strike, time to maturity (years) of each option; arrays
    v0           : initial variance, fixed from step 1
    kappa, theta : speed of mean reversion and long-run variance, fixed from step 1
    option       : 'call' or 'put' for each option; array
    mrk_iv       : market mid implied vol of each option; array

    Returns
    -------
    Array of residuals, one per option.
    """

    eta, rho = params

    res = np.zeros(len(K))

    for i in range(len(K)):

        model_price = heston_price(
            F[i], K[i], tau[i], v0, kappa, theta, eta, rho, option[i]
        )

        model_iv = black_implied_vol(model_price, F[i], K[i], tau[i], option[i])

        res[i] = model_iv - mrk_iv[i]

    return res

# residual with weighted spread
def res_2_weighted(
    params:list, F:np.array, K:np.array, tau:np.array, 
    v0:float, kappa:float, theta:float, option:np.ndarray,
    mrk_iv:np.ndarray, iv_spread:np.ndarray, eps:float=0.0005) -> np.ndarray:
    """
    Residuals for the step-2 weighted fit: implied vol errors divided by the bid-ask spread.

        res_i = [sigma_imp_i(Heston) - sigma_imp_i(market)] / w_i,  w_i = max(eps, ask_iv_i - bid_iv_i)

    least_squares squares the residuals, so each error counts as (error / w_i)^2: tightly quoted
    options get more weight, wide quotes less. eps stops a near-zero spread from dominating.

    Parameters
    ----------
    params       : [eta, rho], the parameters being fitted
    F, K, tau    : forward, strike, time to maturity (years) of each option; arrays
    v0           : initial variance, fixed from step 1
    kappa, theta : speed of mean reversion and long-run variance, fixed from step 1
    option       : 'call' or 'put' for each option; array
    mrk_iv       : market mid implied vol of each option; array
    iv_spread    : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array
    eps          : floor on the weight (default 0.0005 = 0.05 vol points)

    Returns
    -------
    Array of weighted residuals, one per option.
    """

    eta, rho = params

    res = np.zeros(len(K))

    for i in range(len(K)):

        model_price = heston_price(
            F[i], K[i], tau[i], v0, kappa, theta, eta, rho, option[i]
        )

        model_iv = black_implied_vol(model_price, F[i], K[i], tau[i], option[i])

        # weight: max(eps, bid-ask spread of the implied vol)
        weight = max(eps, iv_spread[i])

        # least_squares squares this, giving (model - market)^2 / max(eps, spread)^2
        res[i] = (model_iv - mrk_iv[i]) / weight

    return res

# residual + weighted + feller condition
def res_2_feller(
    params:list, F:np.array, K:np.array, tau:np.array, 
    v0:float, kappa:float, theta:float, option:np.ndarray,
    mrk_iv:np.ndarray, iv_spread:np.ndarray,
    eps:float=0.0005, penalty_weight:float=1e5) -> np.ndarray:
    """
    Residuals for the step-2 Feller fit: weighted implied vol errors plus a Feller penalty.

    The first len(K) residuals are the same as res_2_weighted. One extra residual penalises
    violating the Feller condition 2 * kappa * theta >= eta^2:

        res_last = penalty_weight * max(0, eta^2 - 2 * kappa * theta)

    least_squares squares it, so the penalty added to the objective is
    (penalty_weight * max(0, eta^2 - 2 * kappa * theta))^2.

    Parameters
    ----------
    params         : [eta, rho], the parameters being fitted
    F, K, tau      : forward, strike, time to maturity (years) of each option; arrays
    v0             : initial variance, fixed from step 1
    kappa, theta   : speed of mean reversion and long-run variance, fixed from step 1
    option         : 'call' or 'put' for each option; array
    mrk_iv         : market mid implied vol of each option; array
    iv_spread      : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array
    eps            : floor on the weight (default 0.0005 = 0.05 vol points)
    penalty_weight : size of the Feller penalty (default 1e5)

    Returns
    -------
    Array of len(K) + 1 residuals: one per option, then the Feller penalty.
    """

    eta, rho = params

    # one residual per option, plus one extra slot for the Feller penalty
    res = np.zeros(len(K) + 1)

    # Step 1: spread-weighted IV errors (same as res_2_weighted)
    for i in range(len(K)):

        model_price = heston_price(
            F[i], K[i], tau[i], v0, kappa, theta, eta, rho, option[i]
        )

        model_iv = black_implied_vol(model_price, F[i], K[i], tau[i], option[i])

        weight = max(eps, iv_spread[i])

        res[i] = (model_iv - mrk_iv[i]) / weight

    # Step 2: Feller penalty, which is zero when 2*kappa*theta >= eta^2
    feller_violation = eta**2 - 2 * kappa * theta

    if feller_violation > 0:
        res[-1] = penalty_weight * feller_violation
    else:
        res[-1] = 0.0

    return res

# regular fit
def fit_eta_rho(
    initial_guess:list, F:np.array, K:np.array, tau:np.array, 
    v0:float, kappa:float, theta:float, option:np.ndarray,
    mrk_iv:np.ndarray):
    """
    Step 2 of the calibration, regular fit: fit eta and rho to the smiles, with v0, kappa, theta fixed.

    Solves
        eta*, rho* = argmin sum_i [ sigma_imp_i(Heston) - sigma_imp_i(market) ]^2
    by bounded least squares (residuals from res_2), eta in [0.05, 3] and rho in [-0.95, 0.95].

    Parameters
    ----------
    initial_guess : [eta_0, rho_0]
    F, K, tau     : forward, strike, time to maturity (years) of each option; arrays
    v0            : initial variance, fixed from step 1
    kappa, theta  : speed of mean reversion and long-run variance, fixed from step 1
    option        : 'call' or 'put' for each option; array
    mrk_iv        : market mid implied vol of each option; array

    Returns
    -------
    (eta, rho, sse), where sse is the sum of squared residuals at the optimum.
    """

    eta_0, rho_0 = initial_guess

    result = least_squares(
        fun=res_2,
        x0=[eta_0, rho_0], 
        args = (F, K, tau, v0, kappa, theta, option, mrk_iv),
        bounds=([0.05, -0.95], [3, 0.95])
        )

    eta, rho = result.x

    sse = 2 * result.cost

    return eta, rho, sse

# weighted fit
def fit_eta_rho_weighted(
    initial_guess:list, F:np.array, K:np.array, tau:np.array, 
    v0:float, kappa:float, theta:float, option:np.ndarray,
    mrk_iv:np.ndarray, iv_spread:np.ndarray):
    """
    Step 2 of the calibration, weighted fit: fit eta and rho with errors weighted by the bid-ask spread.

    Solves
        eta*, rho* = argmin sum_i [ (sigma_imp_i(Heston) - sigma_imp_i(market)) / w_i ]^2,
        w_i = max(eps, ask_iv_i - bid_iv_i)
    by bounded least squares (residuals from res_2_weighted), eta in [0.05, 3] and rho in [-0.95, 0.95].

    Parameters
    ----------
    initial_guess : [eta_0, rho_0]
    F, K, tau     : forward, strike, time to maturity (years) of each option; arrays
    v0            : initial variance, fixed from step 1
    kappa, theta  : speed of mean reversion and long-run variance, fixed from step 1
    option        : 'call' or 'put' for each option; array
    mrk_iv        : market mid implied vol of each option; array
    iv_spread     : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array

    Returns
    -------
    (eta, rho, sse), where sse is the weighted sum of squared residuals at the optimum.
    """

    eta_0, rho_0 = initial_guess

    result = least_squares(
        fun=res_2_weighted,
        x0=[eta_0, rho_0], 
        args = (F, K, tau, v0, kappa, theta, option, mrk_iv, iv_spread),
        bounds=([0.05, -0.95], [3, 0.95])
        )

    eta, rho = result.x

    sse = 2 * result.cost

    return eta, rho, sse

# feller fit
def fit_eta_rho_feller(
    initial_guess:list, F:np.array, K:np.array, tau:np.array, 
    v0:float, kappa:float, theta:float, option:np.ndarray,
    mrk_iv:np.ndarray, iv_spread:np.ndarray):
    """
    Step 2 of the calibration, Feller fit: weighted fit of eta and rho with a Feller penalty.

    Solves
        eta*, rho* = argmin { sum_i [ (sigma_imp_i(Heston) - sigma_imp_i(market)) / w_i ]^2
                              + [ penalty_weight * max(0, eta^2 - 2 * kappa * theta) ]^2 }
    by bounded least squares (residuals from res_2_feller), eta in [0.05, 3] and rho in [-0.95, 0.95].
    With kappa and theta fixed, the penalty effectively caps eta at sqrt(2 * kappa * theta).

    Parameters
    ----------
    initial_guess : [eta_0, rho_0]
    F, K, tau     : forward, strike, time to maturity (years) of each option; arrays
    v0            : initial variance, fixed from step 1
    kappa, theta  : speed of mean reversion and long-run variance, fixed from step 1
    option        : 'call' or 'put' for each option; array
    mrk_iv        : market mid implied vol of each option; array
    iv_spread     : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array

    Returns
    -------
    (eta, rho, sse), where sse includes the squared Feller penalty.
    """

    eta_0, rho_0 = initial_guess

    result = least_squares(
        fun=res_2_feller,
        x0=[eta_0, rho_0], 
        args = (F, K, tau, v0, kappa, theta, option, mrk_iv, iv_spread),
        bounds=([0.05, -0.95], [3, 0.95])
        )

    eta, rho = result.x

    sse = 2 * result.cost

    return eta, rho, sse

def add_heston_iv(df:pd.DataFrame, heston_params:list) -> pd.DataFrame:
    """
    Add Heston model price and Heston implied vol columns to a smile dataframe.

    Parameters
    ----------
    df            : dataframe with columns 'F', 'Strike', 'tau', 'type'
    heston_params : [v_0, kappa, theta, eta, rho]   (note: kappa before theta)

    Returns
    -------
    A copy of df with two new columns: 'Heston_price' and 'Heston_iv'
    """

    # Step 1: unpack parameters (order: v_0, kappa, theta, eta, rho)
    v_0, kappa, theta, eta, rho = heston_params

    # Step 2: work on a copy so the input dataframe is not changed
    df = df.copy()

    # Step 3: Heston price for each strike
    df['Heston_price'] = df.apply(
        lambda row: heston_price(
            F=row['F'], K=row['Strike'], tau=row['tau'],
            v0=v_0, kappa=kappa, theta=theta, eta=eta, rho=rho,
            option=row['type']),
        axis=1)

    # Step 4: back out the Black implied vol from the Heston price
    df['Heston_iv'] = df.apply(
        lambda row: black_implied_vol(
            row['Heston_price'], row['F'], row['Strike'], row['tau'], row['type']),
        axis=1)

    return df

def fit_error_summary(df:pd.DataFrame, market_col:str='Mid_iv') -> dict:
    """
    Error statistics between Heston IV and a market IV column, in vol points.

    Parameters
    ----------
    df         : dataframe containing 'Heston_iv' and the market IV column
    market_col : which market IV to compare against: 'Mid_iv', 'Ask_iv' or 'Bid_iv'

    Returns
    -------
    {'RMSE', 'Max abs', 'Mean'} of Heston_iv - market IV, in vol points (1 = 0.01 vol).
    """

    # error for each strike, in vol points (0.01 = 1 vol point)
    error = (df['Heston_iv'] - df[market_col]) * 100

    rmse = np.sqrt((error**2).mean())
    max_abs = error.abs().max()
    mean = error.mean()

    return {'RMSE': rmse, 'Max abs': max_abs, 'Mean': mean}

# Calibrate surface
# regular residual
def res_surface_1(
    params:list, F:np.array, K:np.array, tau:np.array, option:np.ndarray, mrk_iv:np.ndarray) -> np.ndarray:
    """
    Residuals for the final (surface) regular fit: Heston implied vol minus market implied vol,
    with all five parameters free.

        res_i = sigma_imp_i(Heston; v, kappa, theta, eta, rho) - sigma_imp_i(market)

    Parameters
    ----------
    params    : [v, kappa, theta, eta, rho], the parameters being fitted
    F, K, tau : forward, strike, time to maturity (years) of each option; arrays
    option    : 'call' or 'put' for each option; array
    mrk_iv    : market mid implied vol of each option; array

    Returns
    -------
    Array of residuals, one per option.
    """

    v, kappa, theta, eta, rho = params

    res = np.zeros(len(K))

    for i in range(len(K)):

        model_price = heston_price(
            F[i], K[i], tau[i], v, kappa, theta, eta, rho, option[i]
        )

        model_iv = black_implied_vol(model_price, F[i], K[i], tau[i], option[i])

        res[i] = model_iv - mrk_iv[i]

    return res

# regular fit
def fit_surface_1(
    initial_guess:list, F:np.array, K:np.array, tau:np.array, option:np.ndarray,
    mrk_iv:np.ndarray):
    """
    Final calibration, regular fit: fit all five Heston parameters to the full implied vol surface.

    Solves
        v*, kappa*, theta*, eta*, rho* = argmin sum_i [ sigma_imp_i(Heston) - sigma_imp_i(market) ]^2
    by bounded least squares (residuals from res_surface_1), with
        v in [0.0001, 1], kappa in [0.1, 5], theta in [0.0001, 1], eta in [0.05, 3], rho in [-0.95, 0.95].

    Parameters
    ----------
    initial_guess : [v_0, kappa, theta, eta, rho], e.g. the results of steps 1 and 2
    F, K, tau     : forward, strike, time to maturity (years) of each option; arrays
    option        : 'call' or 'put' for each option; array
    mrk_iv        : market mid implied vol of each option; array

    Returns
    -------
    (v, kappa, theta, eta, rho, sse), where sse is the sum of squared residuals at the optimum.
    """

    v, kappa, theta, eta, rho = initial_guess

    result = least_squares(
        fun=res_surface_1,
        x0=[v, kappa, theta, eta, rho], 
        args = (F, K, tau, option, mrk_iv),
        bounds=([0.0001, 0.1, 0.0001, 0.05, -0.95], [1, 5, 1, 3, 0.95])
        )

    v, kappa, theta, eta, rho = result.x

    sse = 2 * result.cost

    return v, kappa, theta, eta, rho, sse

# weighted residual
def res_surface_2(
    params:list, F:np.array, K:np.array, tau:np.array, option:np.ndarray,
    mrk_iv:np.ndarray, iv_spread:np.ndarray, eps:float=0.0005) -> np.ndarray:
    """
    Residuals for the final (surface) weighted fit: implied vol errors divided by the bid-ask spread,
    with all five parameters free.

        res_i = [sigma_imp_i(Heston) - sigma_imp_i(market)] / w_i,  w_i = max(eps, ask_iv_i - bid_iv_i)

    Parameters
    ----------
    params    : [v, kappa, theta, eta, rho], the parameters being fitted
    F, K, tau : forward, strike, time to maturity (years) of each option; arrays
    option    : 'call' or 'put' for each option; array
    mrk_iv    : market mid implied vol of each option; array
    iv_spread : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array
    eps       : floor on the weight (default 0.0005 = 0.05 vol points)

    Returns
    -------
    Array of weighted residuals, one per option.
    """

    v, kappa, theta, eta, rho = params

    res = np.zeros(len(K))

    for i in range(len(K)):

        model_price = heston_price(
            F[i], K[i], tau[i], v, kappa, theta, eta, rho, option[i]
        )

        model_iv = black_implied_vol(model_price, F[i], K[i], tau[i], option[i])

        # weight: max(eps, bid-ask spread of the implied vol)
        weight = max(eps, iv_spread[i])

        # least_squares squares this, giving (model - market)^2 / max(eps, spread)^2
        res[i] = (model_iv - mrk_iv[i]) / weight

    return res

# weighted fit
def fit_surface_2(
    initial_guess:list, F:np.array, K:np.array, tau:np.array, option:np.ndarray,
    mrk_iv:np.ndarray, iv_spread:np.ndarray):
    """
    Final calibration, weighted fit: fit all five Heston parameters with errors weighted by the bid-ask spread.

    Solves
        v*, kappa*, theta*, eta*, rho* = argmin sum_i [ (sigma_imp_i(Heston) - sigma_imp_i(market)) / w_i ]^2,
        w_i = max(eps, ask_iv_i - bid_iv_i)
    by bounded least squares (residuals from res_surface_2), with the same bounds as fit_surface_1.

    Parameters
    ----------
    initial_guess : [v_0, kappa, theta, eta, rho], e.g. the results of steps 1 and 2
    F, K, tau     : forward, strike, time to maturity (years) of each option; arrays
    option        : 'call' or 'put' for each option; array
    mrk_iv        : market mid implied vol of each option; array
    iv_spread     : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array

    Returns
    -------
    (v, kappa, theta, eta, rho, sse), where sse is the weighted sum of squared residuals at the optimum.
    """

    v, kappa, theta, eta, rho = initial_guess

    result = least_squares(
        fun=res_surface_2,
        x0=[v, kappa, theta, eta, rho], 
        args = (F, K, tau, option, mrk_iv, iv_spread),
        bounds=([0.0001, 0.1, 0.0001, 0.05, -0.95], [1, 5, 1, 3, 0.95])
        )

    v, kappa, theta, eta, rho = result.x

    sse = 2 * result.cost

    return v, kappa, theta, eta, rho, sse

# Feller residual
def res_surface_3(
    params:list, F:np.array, K:np.array, tau:np.array, 
    option:np.ndarray, mrk_iv:np.ndarray, iv_spread:np.ndarray,
    eps:float=0.0005, penalty_weight:float=1e5) -> np.ndarray:
    """
    Residuals for the final (surface) Feller fit: weighted implied vol errors plus a Feller penalty,
    with all five parameters free.

    The first len(K) residuals are the same as res_surface_2. One extra residual penalises
    violating the Feller condition 2 * kappa * theta >= eta^2:

        res_last = penalty_weight * max(0, eta^2 - 2 * kappa * theta)

    least_squares squares it, so the penalty added to the objective is
    (penalty_weight * max(0, eta^2 - 2 * kappa * theta))^2.

    Parameters
    ----------
    params         : [v, kappa, theta, eta, rho], the parameters being fitted
    F, K, tau      : forward, strike, time to maturity (years) of each option; arrays
    option         : 'call' or 'put' for each option; array
    mrk_iv         : market mid implied vol of each option; array
    iv_spread      : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array
    eps            : floor on the weight (default 0.0005 = 0.05 vol points)
    penalty_weight : size of the Feller penalty (default 1e5)

    Returns
    -------
    Array of len(K) + 1 residuals: one per option, then the Feller penalty.
    """

    v, kappa, theta, eta, rho = params

    # one residual per option, plus one extra slot for the Feller penalty
    res = np.zeros(len(K) + 1)

    # Step 1: spread-weighted IV errors (same as res_2_weighted)
    for i in range(len(K)):

        model_price = heston_price(
            F[i], K[i], tau[i], v, kappa, theta, eta, rho, option[i]
        )

        model_iv = black_implied_vol(model_price, F[i], K[i], tau[i], option[i])

        weight = max(eps, iv_spread[i])

        res[i] = (model_iv - mrk_iv[i]) / weight

    # Step 2: Feller penalty, which is zero when 2*kappa*theta >= eta^2
    feller_violation = eta**2 - 2 * kappa * theta

    if feller_violation > 0:
        res[-1] = penalty_weight * feller_violation
    else:
        res[-1] = 0.0

    return res


# Feller fit
def fit_surface_3(
    initial_guess:list, F:np.array, K:np.array, tau:np.array,
    option:np.ndarray, mrk_iv:np.ndarray, iv_spread:np.ndarray):
    """
    Final calibration, Feller fit: weighted fit of all five Heston parameters with a Feller penalty.

    Solves
        v*, kappa*, theta*, eta*, rho* = argmin { sum_i [ (sigma_imp_i(Heston) - sigma_imp_i(market)) / w_i ]^2
                                                  + [ penalty_weight * max(0, eta^2 - 2 * kappa * theta) ]^2 }
    by bounded least squares (residuals from res_surface_3), with the same bounds as fit_surface_1.

    Parameters
    ----------
    initial_guess : [v_0, kappa, theta, eta, rho], e.g. the results of steps 1 and 2
    F, K, tau     : forward, strike, time to maturity (years) of each option; arrays
    option        : 'call' or 'put' for each option; array
    mrk_iv        : market mid implied vol of each option; array
    iv_spread     : implied vol bid-ask spread (ask_iv - bid_iv) of each option; array

    Returns
    -------
    (v, kappa, theta, eta, rho, sse), where sse includes the squared Feller penalty.
    """

    v, kappa, theta, eta, rho = initial_guess

    result = least_squares(
        fun=res_surface_3,
        x0=[v, kappa, theta, eta, rho], 
        args = (F, K, tau, option, mrk_iv, iv_spread),
        bounds=([0.0001, 0.1, 0.0001, 0.05, -0.95], [1, 5, 1, 3, 0.95])
        )

    v, kappa, theta, eta, rho = result.x

    sse = 2 * result.cost

    return v, kappa, theta, eta, rho, sse