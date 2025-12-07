import gradio as gr
import pandas as pd
from datetime import datetime, timedelta
import analysis
import plots

def run_analysis(ticker, start_date, end_date, forecast_years, extra_percentiles_str, initial_investment, simulations):
    """
    Main handler for the Gradio interface.
    """
    if not ticker:
        return "Please enter a ticker symbol.", None, None, None, None
    
    # 1. Fetch Data
    df, error = analysis.fetch_data(ticker, start_date, end_date)
    if error:
        return f"Error fetching data: {error}", None, None, None, None
        
    if len(df) < 30:
        return "Insufficient data points for analysis (need at least 30).", None, None, None, None
    
    # 2. Calculate Returns
    returns_dict = analysis.calculate_returns(df)
    daily_returns = returns_dict['daily']
    
    # 3. Fit Distribution
    best_dist, best_params, fit_results = analysis.get_best_fit_distribution(daily_returns)
    fit_summary = f"Best Fit Distribution: {best_dist.name}\n"
    
    # 4. Simulation
    # Use Initial Investment as the starting price for the simulation to show Portfolio Value
    start_value = initial_investment if initial_investment else 10000.0
    num_simulations = int(simulations) if simulations else 5000
    
    last_price = df['Adj Close'].iloc[-1] if 'Adj Close' in df.columns else df['Close'].iloc[-1]
    days_to_simulate = int(forecast_years * 252)
    
    simulation_paths = analysis.monte_carlo_simulation(start_value, best_dist, best_params, days_to_simulate, simulations=num_simulations)
    
    # 5. Percentiles
    percentiles_list = [1, 5, 10, 25, 50, 75, 90]
    
    # Parse user extra percentiles
    if extra_percentiles_str:
        try:
            extras = [float(p.strip()) for p in extra_percentiles_str.split(',') if p.strip()]
            percentiles_list.extend(extras)
            percentiles_list = sorted(list(set(percentiles_list))) # Unique and sorted
        except ValueError:
            pass # Ignore invalid inputs
            
    percentile_values = analysis.calculate_percentiles(simulation_paths, percentiles_list)
    
    # Prepare Outputs
    
    # Text Report
    report = f"Analysis for {ticker}\n"
    report += f"Data Range: {start_date} to {end_date}\n"
    report += f"Last Price: {last_price:.2f}\n"
    report += f"Initial Investment: ${start_value:,.2f}\n"
    report += f"Best Fit Distribution: {best_dist.name}\n\n"
    report += f"Projected Portfolio Value (End of {forecast_years} Years):\n"
    for p, val in percentile_values.items():
        report += f"{p}th Percentile: ${val:,.2f}\n"
        
    # Charts
    chart_price = plots.plot_historical_prices(df, ticker)
    chart_dist = plots.plot_returns_distribution(daily_returns, best_dist, best_params)
    chart_sim = plots.plot_monte_carlo(simulation_paths, percentile_values)
    
    return report, chart_price, chart_dist, chart_sim

# Build UI
with gr.Blocks(title="Equity Analysis App") as app:
    gr.Markdown("# Comprehensive Equity Analysis")
    gr.Markdown("Analyze total returns, fit statistical distributions, and run Monte Carlo simulations.")
    
    with gr.Row():
        with gr.Column(scale=1):
            ticker_input = gr.Textbox(label="Ticker Symbol", value="SPY", placeholder="e.g. AAPL, MSFT, SPY")
            
            # Date defaults
            default_end = datetime.now().strftime("%Y-%m-%d")
            default_start = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
            
            start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=default_start)
            end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)", value=default_end)
            
            initial_investment_input = gr.Number(label="Initial Investment ($)", value=10000)
            years_input = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Forecast Horizon (Years)")
            simulations_input = gr.Slider(minimum=100, maximum=10000, value=5000, step=100, label="Simulation Runs")
            
            percentiles_input = gr.Textbox(label="Additional Percentiles (comma separated)", placeholder="e.g. 2.5, 97.5")
            
            run_btn = gr.Button("Run Analysis", variant="primary")
            
        with gr.Column(scale=3):
            output_report = gr.Textbox(label="Analysis Report", lines=12)
            
    with gr.Row():
        chart_price_output = gr.Plot(label="Historical Price")
        
    with gr.Row():
        chart_dist_output = gr.Plot(label="Return Distribution Fit")
        chart_sim_output = gr.Plot(label="Monte Carlo Simulation")

    run_btn.click(
        fn=run_analysis,
        inputs=[ticker_input, start_date_input, end_date_input, years_input, percentiles_input, initial_investment_input, simulations_input],
        outputs=[output_report, chart_price_output, chart_dist_output, chart_sim_output]
    )

if __name__ == "__main__":
    app.launch()
