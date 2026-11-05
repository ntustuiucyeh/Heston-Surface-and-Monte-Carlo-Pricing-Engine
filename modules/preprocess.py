"""
Option data preparation: file reading, quote cleaning, mid prices, discount factors from put-call
parity, and Black implied volatilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq

# initialize data path
def set_data_path(data_folder_name:str) -> Path:
    """
    Build the path of the data folder inside the current working directory, and print both paths.

    Parameters
    ----------
    data_folder_name : name of the data folder (e.g. 'Data')

    Returns
    -------
    (current_dr_path, data_dr_path) as Path objects.
    """

    current_dr_path = Path.cwd()
    data_dr_path = current_dr_path / data_folder_name

    print(f'Current Directory Path: {current_dr_path}')
    print(f'Data Folder Path: {data_dr_path}')

    return current_dr_path, data_dr_path

# read files
def read_file(folder_path: Path, 
              file_name: str, 
              read_function:function,
              dispaly_path:bool = True,
              **kwarg) -> pd.DataFrame:
    """
    Read a data file into a DataFrame.

    Parameters
    ----------
    folder_path   : folder that contains the file
    file_name     : file name (e.g. 'Surface.csv')
    read_function : pandas reader to use (e.g. pd.read_csv, pd.read_excel)
    dispaly_path  : print the full file path if True
    **kwarg       : extra arguments passed to read_function (e.g. header, index_col)

    Returns
    -------
    The file contents as a DataFrame.
    """

    file_path = folder_path / file_name
    source = read_function(file_path, **kwarg)
    df = pd.DataFrame(source)

    if dispaly_path:
        print(f'{file_name}\'s Path: {file_path}')

    return df 

# dataframe info
def df_info(df:pd.DataFrame) -> None:
    """
    Print the shape, columns, index names and first rows of a DataFrame.

    Parameters
    ----------
    df : DataFrame to inspect

    Returns
    -------
    None
    """

    print(f'Dimension: {df.shape}\n')
    print(f'Columns:{df.columns}\n')
    print(f'Indexs:{df.index.names}\n')
    print(df.head())

    return None

# backout discount factor
def get_df(call_price:float, put_price:float, strike:float, foward_price:float) -> float:
    """
    Discount factor implied by put-call parity at one strike.

        C - P = DF * (F - K)   =>   DF = (C - P) / (F - K)

    Parameters
    ----------
    call_price   : call price at strike K
    put_price    : put price at strike K
    strike       : strike K
    foward_price : forward price F of the expiry

    Returns
    -------
    Discount factor to the expiry. Unstable near K = F, where F - K is close to zero.
    """

    discount_factor = (call_price - put_price) / (foward_price - strike)

    return discount_factor

# Black-Shcoles formula (forward representation)
def bs_price(F:float, K:float, T:float, df:float, sigma:float, option="call") -> float:
    """
    Discounted Black (forward) price of a European call or put.

        Call = DF * [F N(d1) - K N(d2)],   Put = DF * [K N(-d2) - F N(-d1)]
        d1 = [ln(F/K) + 0.5 sigma^2 T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)

    Parameters
    ----------
    F      : forward price
    K      : strike
    T      : time to expiry (years)
    df     : discount factor to the expiry
    sigma  : volatility
    option : 'call' or 'put'

    Returns
    -------
    Option price today (None if option is neither 'call' nor 'put').
    """
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))

    elif option == "put":
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

# backout implied volatility
def get_iv(market_price:float, F:float, K:float, T:float, df:float, option="call") -> float:
    """
    Black implied volatility of one option price.

    Finds sigma such that bs_price(F, K, T, df, sigma, option) = market_price by Brent's method on
    [1e-8, 5]. Raises ValueError if no volatility in that range reproduces the price.

    Parameters
    ----------
    market_price : option price today
    F            : forward price
    K            : strike
    T            : time to expiry (years)
    df           : discount factor to the expiry
    option       : 'call' or 'put'

    Returns
    -------
    Implied volatility.
    """

    objective = lambda sigma: (
        bs_price(F, K, T, df, sigma, option) - market_price
    )

    iv = brentq(objective, 1e-8, 5.0)

    return iv

# clean low quality data
def clean_smile(df:pd.DataFrame,
                bid_threshold:float,
                volumn_threshold:int,
                option_type: str = 'Call') -> pd.DataFrame:
    """
    Remove low-quality quotes: a bid equal to bid_threshold (e.g. zero) or a volume below volumn_threshold.

    Parameters
    ----------
    df               : option chain with two-level columns (option type, field), e.g. ('Call', 'Bid')
                       and ('Put', 'Volm')
    bid_threshold    : bid value to drop (e.g. 0 for no bid)
    volumn_threshold : minimum traded volume to keep
    option_type      : 'Call' or 'Put' to clean and return that side only, or 'Both' to drop a strike
                       if either side fails

    Returns
    -------
    Filtered DataFrame: both sides for 'Both', otherwise the chosen side only.
    """

    if option_type == 'Both':
        bid_mask = df.loc[:, pd.IndexSlice[:,'Bid']].eq(bid_threshold).any(axis=1)
        vol_mask = df.loc[:, pd.IndexSlice[:,'Volm']].lt(volumn_threshold).any(axis=1)
        smile_df = df[~(bid_mask | vol_mask)]

        return smile_df

    else:
        option_df = df.loc[:, pd.IndexSlice[option_type]]
        bid_mask = option_df.loc[:, 'Bid'].eq(bid_threshold)
        volum_mask = option_df.loc[:, 'Volm'].lt(volumn_threshold)
        smile_call_df = option_df[~(bid_mask | volum_mask)] 

        return smile_call_df

# compute mid price
def mid_price(df:pd.DataFrame, both_option:bool = True) -> None:
    """
    Add mid prices, (ask + bid) / 2, to the DataFrame in place.

    Parameters
    ----------
    df          : option quotes
    both_option : True for two-level columns, adding ('Call', 'Mid') and ('Put', 'Mid');
                  False for single-level columns, adding 'Mid'

    Returns
    -------
    None (df is modified in place).
    """

    if both_option:
        for option in ['Call', 'Put']:

            df[(option, 'Mid')] = (df.loc[:, pd.IndexSlice[option, 'Ask']] + df.loc[:, pd.IndexSlice[option, 'Bid']]) / 2

    else:

        df['Mid'] = (df['Ask'] + df['Bid']) / 2

    return None

# compute year fraction
def year_fraction(t0:pd.Timestamp, df:pd.DataFrame) -> None:
    """
    Add a 'T' column with the time to expiry in years (ACT/365), in place.

    Parameters
    ----------
    t0 : valuation date
    df : DataFrame with an 'Expiry' index level

    Returns
    -------
    None (df is modified in place).
    """

    expiry = df.index.get_level_values("Expiry")
    df['T'] = (expiry - t0).days / 365

    return None

# backout discount factor for call and put dataframe
def get_df_both(df_raw:pd.DataFrame, f_list:list) -> dict:
    """
    Discount factor for each expiry from put-call parity, using mid, ask and bid prices.

    At every strike, DF = (C - P) / (F - K) is computed from the mid, ask and bid prices; for each expiry
    the median over strikes is taken, which is robust to outlying strikes.

    Parameters
    ----------
    df_raw : option chain indexed by (Expiry, Strike), with ('Call', 'Mid' / 'Ask' / 'Bid') and
             ('Put', 'Mid' / 'Ask' / 'Bid') columns; not modified
    f_list : forward price of each expiry, in the order the expiries appear in the index

    Returns
    -------
    Dict {expiry: (df_mid, df_ask, df_bid)}.
    """

    df = df_raw.copy()

    expiry_list = df.index.get_level_values('Expiry').unique().to_list()

    dic = dict(zip(expiry_list, f_list))

    result = []

    for expiry, f in dic.items():
        
        df.loc[expiry, ('DF Mid', '')] = df.loc[expiry].apply(lambda row: 
            get_df(row[("Call", "Mid")], row[("Put", "Mid")], row.name, f), axis=1).values    
        
        df.loc[expiry, ('DF Ask', '')] = df.loc[expiry].apply(lambda row: 
            get_df(row[("Call", "Ask")], row[("Put", "Ask")], row.name, f), axis=1).values
        
        df.loc[expiry, ('DF Bid', '')] = df.loc[expiry].apply(lambda row: 
            get_df(row[("Call", "Bid")], row[("Put", "Bid")], row.name, f), axis=1).values

        df_mid = df.loc[expiry, ('DF Mid', '')].median()
        df_ask = df.loc[expiry, ('DF Ask', '')].median()
        df_bid = df.loc[expiry, ('DF Bid', '')].median()

        df_set = (df_mid, df_ask, df_bid)

        result.append(df_set)

    result_dic = dict(zip(expiry_list, result))
    
    return result_dic

# backout implied volatility for whole dataframe
def get_all_iv(df:pd.DataFrame, f_list:list, DF_dict: dict, option = 'call') -> None:
    """
    Add Black implied volatilities from the mid, ask and bid prices, and the bid-ask IV spread, in place.

    Adds the columns 'Mid_iv', 'Ask_iv', 'Bid_iv' and 'iv_spread' (= Ask_iv - Bid_iv). All three use the
    mid discount factor of each expiry, so the spread reflects the price quotes only.

    Parameters
    ----------
    df      : quotes for one option type, indexed by (Expiry, Strike), with 'Mid', 'Ask', 'Bid' and 'T'
              columns
    f_list  : forward price of each expiry, in the same order as DF_dict
    DF_dict : {expiry: (df_mid, df_ask, df_bid)} from get_df_both
    option  : 'call' or 'put'

    Returns
    -------
    None (df is modified in place).
    """

    # forming loop dictionary
    loop_dict = {}
    for i, (expiry, df_triplet) in enumerate(DF_dict.items()):

        duadruple = df_triplet + (f_list[i],)

        loop_dict[expiry] = duadruple

    for expiry, (df_mid, df_ask, df_bid, forward) in loop_dict.items():

        df.loc[expiry, 'Mid_iv'] = df.loc[expiry].apply(lambda row: 
            get_iv(row['Mid'], forward, row.name, row['T'], df_mid, option), axis=1).values

        df.loc[expiry, 'Ask_iv'] = df.loc[expiry].apply(lambda row: 
            get_iv(row['Ask'], forward, row.name, row['T'], df_mid, option), axis=1).values
        
        df.loc[expiry, 'Bid_iv'] = df.loc[expiry].apply(lambda row: 
            get_iv(row['Bid'], forward, row.name, row['T'], df_mid, option), axis=1).values
        
        df.loc[expiry, 'iv_spread'] = (df.loc[expiry, 'Ask_iv'] - df.loc[expiry, 'Bid_iv']).values

    return None

# display ITM, OTM, ALL descriptive statistics
def discrptive_stat(
    df:pd.DataFrame, forward:float, col_name:str, expiray:str, option = 'call') -> pd.DataFrame:
    """
    Descriptive statistics of one column for one expiry: all strikes, OTM strikes and ITM strikes.

    Calls are OTM above the forward, puts below it.

    Parameters
    ----------
    df       : option data indexed by (Expiry, Strike)
    forward  : forward price of the expiry
    col_name : column to describe (e.g. 'Mid_iv', 'iv_spread')
    expiray  : expiry to select
    option   : 'call' or 'put'

    Returns
    -------
    DataFrame with columns 'All <col_name>', 'OTM <col_name>' and 'ITM <col_name>' (count, mean, std,
    min, quartiles, max).
    """

    if option == 'call':
        all_stat = df.loc[expiray, col_name].describe()
        otm_stat = df.loc[expiray, col_name][df.loc[expiray].index > forward].describe()
        itm_stat = df.loc[expiray, col_name][df.loc[expiray].index <= forward].describe()

    else:
        all_stat = df.loc[expiray, col_name].describe()
        otm_stat = df.loc[expiray, col_name][df.loc[expiray].index < forward].describe()
        itm_stat = df.loc[expiray, col_name][df.loc[expiray].index >= forward].describe()

    summary = pd.concat(
        [all_stat, otm_stat, itm_stat], axis=1, keys = [f'All {col_name}', f'OTM {col_name}', f'ITM {col_name}'])

    return summary
