import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as st
from datetime import datetime, timedelta

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical data for the given ticker.
    Focuses on 'Adj Close' to account for dividends and splits (Total Return).
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None, "No data found for this ticker and date range."
        
        # yfinance returns a MultiIndex columns if multiple tickers, but we expect one.
        # Ensure we are working with a single level DataFrame or Series
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
            
        return df, None
    except Exception as e:
        return None, str(e)

def calculate_returns(df):
    """
    Calculates Daily, Monthly, Quarterly, and Yearly total returns.
    """
    if 'Adj Close' not in df.columns:
         # Fallback if Adj Close is missing (rare for yfinance)
         price_col = 'Close'
    else:
         price_col = 'Adj Close'

    # Daily Return
    daily_returns = df[price_col].pct_change().dropna()

    # Resample for other periods
    # We use 'ME', 'QE', 'YE' for Month, Quarter, Year End in recent pandas versions
    # or 'M', 'Q', 'Y' for older ones. Let's stick to standard alias with checking.
    
    monthly_prices = df[price_col].resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    
    quarterly_prices = df[price_col].resample('QE').last()
    quarterly_returns = quarterly_prices.pct_change().dropna()
    
    yearly_prices = df[price_col].resample('YE').last()
    yearly_returns = yearly_prices.pct_change().dropna()
    
    return {
        'daily': daily_returns,
        'monthly': monthly_returns,
        'quarterly': quarterly_returns,
        'yearly': yearly_returns
    }

def get_best_fit_distribution(data, distributions=None):
    """
    Models equity daily returns using common statistical distributions
    and finds the one with the best fit (using Sum of Squared Errors).
    """
    if distributions is None:
        distributions = [
            st.norm,
            st.t,
            st.lognorm,
            st.nct,
            st.gennorm,
            st.laplace,
            st.expon,
            st.cauchy,
            st.skewcauchy,
            st.skewnorm,
            st.gamma,
            st.beta
        ]
        
    # Get histogram of original data
    y, x = np.histogram(data, bins='auto', density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    
    results = []

    for distribution in distributions:
        try:
            # Fit distribution to data
            params = distribution.fit(data)

            # Separate parts of parameters:
            #   arg = shape parameters
            #   loc = location
            #   scale = scale
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))
            
            results.append((distribution.name, sse, params))

            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse

        except Exception:
            pass

    return best_distribution, best_params, results

def monte_carlo_simulation(last_price, distribution, params, days, simulations=1000):
    """
    Creates a Monte Carlo simulation.
    """
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    # Generate random returns based on the fitted distribution
    # We generate a (simulations, days) matrix
    simulated_returns = distribution.rvs(*arg, loc=loc, scale=scale, size=(simulations, days))
    
    # Calculate price paths
    # Price_t = Price_{t-1} * (1 + Return_t)
    # Price_t = Price_0 * Product(1 + Return_i)
    
    # Add 1 to returns to get growth factors
    growth_factors = 1 + simulated_returns
    
    # Cumulative product along the time axis (axis 1)
    cumulative_growth = np.cumprod(growth_factors, axis=1)
    
    # Multiply by initial price to get price paths
    simulation_paths = last_price * cumulative_growth
    
    return simulation_paths

def calculate_percentiles(simulation_paths, percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99]):
    """
    Calculates percentiles of the final prices from the simulation.
    """
    final_prices = simulation_paths[:, -1]
    results = {p: np.percentile(final_prices, p) for p in percentiles}
    return results

def calculate_drawdown(df):
    """
    Calculates the drawdown series for the given dataframe.
    """
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    else:
        price_col = 'Close'
        
    prices = df[price_col]
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown
