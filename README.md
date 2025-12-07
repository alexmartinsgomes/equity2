# Equity Analysis Application

A comprehensive equity analysis tool built with Python, Gradio, and Plotly. This application allows users to analyze historical total returns, fit statistical distributions to daily returns, and perform Monte Carlo simulations to project future portfolio values.

## Features

- **Total Return Analysis**: 
    - Fetches historical data using `yfinance`.
    - Automatically adjusts for dividends and splits (uses `Adj Close`).
    - Calculates Daily, Monthly, Quarterly, and Yearly returns.
    - **Interactive Chart options**: Toggle Linear/Log scale and Price/Percentage view.
    - **Drawdown Analysis**: Visualizes maximum drawdown and historical drawdowns.

- **Statistical Distribution Fitting**:
    - Tests multiple distributions (Normal, T, LogNormal, etc.) against daily returns.
    - Identifies the best-fit distribution using Sum of Squared Errors (SSE).
    - Visualizes the fit with a probability density histogram.

- **Monte Carlo Simulation**:
    - Simulates future price paths based on the best-fit statistical distribution.
    - **Interactive Features**:
        - **Forecast Horizon**: Customizable time horizon (in years).
        - **Initial Investment**: Project the value of a specific investment amount.
        - **Simulation Runs**: Configurable number of Monte Carlo iterations.
        - **Percentiles**: Customizable percentiles for risk analysis.
    - **Visualization**:
        - Professional, publication-quality charts (Plotly).
        - Distinct lines and colors for selected percentiles.
        - X-axis in Years for clear long-term visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alexmartinsgomes/equity2.git
   cd equity2
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   # Using uv
   uv venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # OR using uv
   uv pip install -r requirements.txt
   ```

## Usage

Run the Gradio application:

```bash
uv run python app.py
# or
python app.py
```

Open your browser at the local URL provided (usually `http://127.0.0.1:7860`).

### User Inputs

1. **Ticker Symbol**: Enter a valid Yahoo Finance ticker (e.g., `SPY`, `AAPL`, `MSFT`).
2. **Date Range**: Select start and end dates for historical analysis.
3. **Initial Investment**: Amount to simulate growing over time.
4. **Forecast Horizon**: Number of years to simulate into the future.
5. **Simulation Runs**: Number of Monte Carlo paths to generate (higher = more precision but slower).
6. **Additional Percentiles**: Add custom percentiles (e.g., `0.1, 99.9`) to the analysis.

## Technologies

- **UI**: Gradio
- **Data**: yfinance, pandas, numpy
- **Statistics**: scipy, statsmodels
- **Visualization**: Plotly

## License

MIT

## Author

Alexandre Martins - martins.gomes.alex@gmail.com