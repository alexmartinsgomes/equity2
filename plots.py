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

def plot_historical_prices(df, ticker, log_scale=False, as_percentage=False):
    """
    Plots historical adjusted close prices.
    """
    if 'Adj Close' in df.columns:
        y_col = 'Adj Close'
    else:
        y_col = 'Close'
    
    y_data = df[y_col]
    
    title_suffix = ""
    y_axis_title = "Price"
    
    if as_percentage:
        # Normalize to start at 0%
        y_data = (y_data / y_data.iloc[0] - 1) * 100
        title_suffix = " (%)"
        y_axis_title = "Return (%)"
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=y_data, 
        mode='lines', 
        name=ticker,
        line=dict(color=COLOR_PRIMARY, width=2)
    ))
    
    if log_scale:
        fig.update_yaxes(type="log")
        title_suffix += " (Log Scale)"

    return apply_professional_layout(fig, f"{ticker} - Historical Total Return{title_suffix}", "Date", y_axis_title)

def plot_drawdown(drawdown_series):
    """
    Plots the drawdown over time.
    """
    # Convert to percentage
    drawdown_pct = drawdown_series * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_pct.index,
        y=drawdown_pct,
        mode='lines',
        name='Drawdown',
        line=dict(color=COLOR_SECONDARY, width=1),
        fill='tozeroy',
        fillcolor='rgba(163, 0, 0, 0.2)' # Transparent Red
    ))
    
    return apply_professional_layout(fig, "Drawdown (%)", "Date", "Drawdown from Peak (%)")

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
    # Convert days to years for X axis
    x_axis = np.arange(days) / 252.0
    
    fig = go.Figure()

    # Plot Median (50th percentile)
    median_path = np.median(simulation_paths, axis=0)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=median_path,
        mode='lines',
        name='Median (50th)',
        line=dict(color=COLOR_PRIMARY, width=3)
    ))
    
    # Calculate and Plot Percentiles with specific colors
    # We will pick a few key ones to highlight if they exist in the data
    # or just plot all requested percentiles that are not median
    
    # Sort percentiles to color gradient or specific assignment
    sorted_ps = sorted([p for p in percentiles_data.keys() if p != 50])
    
    # Generate colors (heatmap style from red to green or similar? Or just distinct)
    # Let's use a spectral palette or distinct colors
    colors = px.colors.qualitative.Bold
    
    for i, p in enumerate(sorted_ps):
        # Calculate path for this percentile over time
        # This is strictly not correct mathematically for "path of the p-th percentile" 
        # vs "p-th percentile of final values", but standard convention is to show 
        # the p-th percentile slice at each time step.
        path_p = np.percentile(simulation_paths, p, axis=0)
        
        # Color cycle
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=path_p,
            mode='lines',
            name=f'{p}th Percentile',
            line=dict(color=color, width=2, dash='dot')
        ))

    # Add text annotations for final values of key percentiles
    max_year = x_axis[-1]
    for p, val in percentiles_data.items():
        if p in [1, 10, 50, 90, 99]: # Label extremes and median
            fig.add_annotation(
                x=max_year, y=val,
                text=f"{p}th: {val:,.0f}",
                showarrow=True,
                arrowhead=1,
                ax= 40,
                ay=0
            )

    return apply_professional_layout(fig, f"Monte Carlo Simulation ({simulations} runs)", "Years into Future", "Projected Portfolio Value")
