import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import scipy.stats as st

# Color palette inspired by financial publications
COLOR_PRIMARY = '#005a9c' # Blue
COLOR_SECONDARY = '#a30000' # Red
COLOR_ACCENT = '#007f3d' # Green
COLOR_GRID = '#e0e0e0'
BACKGROUND_COLOR = '#ffffff'
FONT_FAMILY = "Arial, sans-serif"

def apply_professional_layout(fig, title, x_title, y_title):
    """
    Applies a professional, WSJ-like style to the Plotly figure.
    """
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'family': FONT_FAMILY}
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font={'family': FONT_FAMILY, 'color': '#333333'},
        xaxis={
            'gridcolor': COLOR_GRID,
            'showline': True,
            'linecolor': '#333333'
        },
        yaxis={
            'gridcolor': COLOR_GRID,
            'showline': True,
            'linecolor': '#333333'
        },
        hovermode='x unified',
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_historical_prices(df, ticker):
    """
    Plots historical adjusted close prices.
    """
    if 'Adj Close' in df.columns:
        y_col = 'Adj Close'
    else:
        y_col = 'Close'
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[y_col], 
        mode='lines', 
        name=ticker,
        line=dict(color=COLOR_PRIMARY, width=2)
    ))
    
    return apply_professional_layout(fig, f"{ticker} - Historical Total Return (Adj Close)", "Date", "Price")

def plot_returns_distribution(data, best_distribution, best_params):
    """
    Plots histogram of daily returns overlayed with the best fit distribution PDF.
    """
    # Histogram
    fig = go.Figure()
    
    # Histogram of actual data
    fig.add_trace(go.Histogram(
        x=data,
        histnorm='probability density',
        name='Actual Returns',
        marker_color=COLOR_PRIMARY,
        opacity=0.7,
        nbinsx=50
    ))
    
    # Generate x values for PDF
    x_min, x_max = data.min(), data.max()
    x_range = np.linspace(x_min, x_max, 500)
    
    # Calculate PDF of best fit
    arg = best_params[:-2]
    loc = best_params[-2]
    scale = best_params[-1]
    pdf = best_distribution.pdf(x_range, loc=loc, scale=scale, *arg)
    
    # Line for Best Fit
    fig.add_trace(go.Scatter(
        x=x_range,
        y=pdf,
        mode='lines',
        name=f'Best Fit ({best_distribution.name})',
        line=dict(color=COLOR_SECONDARY, width=3)
    ))

    # Also add Normal distribution for comparison if best is not Normal
    if best_distribution.name != 'norm':
         norm_params = st.norm.fit(data)
         norm_pdf = st.norm.pdf(x_range, *norm_params)
         fig.add_trace(go.Scatter(
            x=x_range,
            y=norm_pdf,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='gray', width=2, dash='dash')
        ))

    return apply_professional_layout(fig, "Daily Returns Distribution & Fit", "Daily Return", "Density")

def plot_monte_carlo(simulation_paths, percentiles_data):
    """
    Plots the results of the Monte Carlo simulation.
    Shows the median path, and shaded areas for various percentiles.
    """
    simulations, days = simulation_paths.shape
    x_axis = np.arange(days)
    
    fig = go.Figure()

    # Plot a few random individual paths (limit to 20 to avoid clutter)
    for i in range(min(20, simulations)):
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=simulation_paths[i, :],
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Plot Median (50th percentile)
    median_path = np.median(simulation_paths, axis=0)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=median_path,
        mode='lines',
        name='Median (50th)',
        line=dict(color=COLOR_PRIMARY, width=3)
    ))
    
    # Plot Confidence Intervals (e.g. 10th-90th, 1st-99th)
    # We can calculate these path-wise for the funnel effect
    p10 = np.percentile(simulation_paths, 10, axis=0)
    p90 = np.percentile(simulation_paths, 90, axis=0)
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=p90,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='90th Percentile'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=p10,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 90, 156, 0.2)', # Transparent Blue
        name='10th-90th Percentile'
    ))

    # Add text annotations for final values of key percentiles
    max_day = days - 1
    for p, val in percentiles_data.items():
        if p in [1, 50, 99]: # Label extremes and median
            fig.add_annotation(
                x=max_day, y=val,
                text=f"{p}th: {val:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=30 if p < 50 else 30,
                ay=0
            )

    return apply_professional_layout(fig, f"Monte Carlo Simulation ({simulations} runs)", "Days into Future", "Projected Price")
