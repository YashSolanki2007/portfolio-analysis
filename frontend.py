'''
The following file is a production version of the main.ipynb file in the form of a dashboard and an analytics report.
'''


import streamlit as st
import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


pd.options.plotting.backend = "plotly"



st.title('Portfolio Analysis')
st.markdown("Portfolio Analysis is a free tool that you can use to analyze and improve your portfolio.")

stocks_inp = st.text_input('Enter tickers of the Portfolio seperated by commas', value='GOOG')
compute_period = int(st.text_input("Enter the time period that you want to track for", value="252"))


stocks_inp = stocks_inp.replace(" ", "")
stocks_inp += ","
stocks = []
curr_ticker = ""
for i in range(len(stocks_inp)):
    if stocks_inp[i] != ",":
        curr_ticker += stocks_inp[i]
    else:
        stocks.append(curr_ticker)
        curr_ticker = ""

TOTAL_DAYS = compute_period
close_prices = []

for i in range(len(stocks)):
    df = si.get_data(stocks[i])
    close_vals = df['close'].to_list()
    close_vals = close_vals[len(close_vals) - TOTAL_DAYS: len(close_vals)]
    close_prices.append(close_vals)
    close_vals = []


close_prices = np.array(close_prices).T
portfolio_df = pd.DataFrame(close_prices, columns=[i for i in stocks])

st.markdown("### Close prices of portfolio")

fig = portfolio_df.plot.line(title="Close prices of portfolio assets", labels={
    "index": "Days", 
    "value": "Price", 
    "variable": "Legend"
})
st.plotly_chart(fig, use_container_width=True)


log_returns = []
for i in portfolio_df.columns:
    stock_log_returns = np.log(
        portfolio_df[i]/portfolio_df[i].shift(1)).dropna().to_list()
    log_returns.append(stock_log_returns)

log_returns = np.array(log_returns).T
log_returns = pd.DataFrame(log_returns, columns=[
                           i for i in portfolio_df.columns])


st.markdown("### Log returns of portfolio assets")
for i in log_returns.columns:
    fig = log_returns[i].plot.hist(labels={"value": "Returns", "variable": "Legend", "count": "frequency"}, 
                                   title=f'Log returns of: {i}')
    st.plotly_chart(fig, use_container_width=True)



st.markdown("### Box plot of log returns")
for i in log_returns.columns:
    fig = log_returns[i].plot.box(labels={"value": "", "variable": i},
                                  title=f'Box and whisker plot of: {i}')
    st.plotly_chart(fig, use_container_width=True)



st.markdown("### Correlation Matrix for the various portfolio assets")
correlation_matrix = portfolio_df.corr()
fig = plt.figure()
ax = sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix for portfolio")
st.pyplot(fig)


st.markdown("### Volitality for the various portfolio assets")
TRADING_DAYS = 60
close_prices_df = pd.DataFrame(close_prices, columns=[i for i in stocks])
volitality = log_returns.rolling(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)
volitality = volitality.dropna()
fig = volitality.plot.line(title=f"Volitality of portfolio assets with {TRADING_DAYS} as period", labels={
    "index": "Days",
    "value": "Volitality",
    "variable": "Legend"
})
st.plotly_chart(fig, use_container_width=True)


st.markdown("### Sharpe Ratio for various portfolio assets")
risk_free_return = 0
sharpe_ratio = (log_returns.rolling(TRADING_DAYS).mean() -
                risk_free_return) / volitality
sharpe_ratio = sharpe_ratio.dropna()

# Annualized sharpe ratio = daily sharpe ratio * sqrt(total_days)
annualized_sharpe_ratio = (
    (log_returns.mean() - risk_free_return) / log_returns.std()) * np.sqrt(TOTAL_DAYS)
fig = sharpe_ratio.plot.line(title=f"Sharpe Ratio of portfolio assets with {TRADING_DAYS} as period", labels={
    "index": "Days",
    "value": "Sharpe Ratio",
    "variable": "Legend"
})
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Annualized Sharpe ratio for the stocks in the portfolio")
for i in range(len(annualized_sharpe_ratio)):
    st.write(
        f"{stocks[i]} has a annualized sharpe ratio of: {annualized_sharpe_ratio[i]}")

st.title("")
st.markdown("### Sortino Ratio for various portfolio assets")
downside_volitality = log_returns[log_returns < 0].rolling(
    TRADING_DAYS, center=True, min_periods=10).std() * np.sqrt(TRADING_DAYS)
sortino_ratio = (log_returns.rolling(TRADING_DAYS).mean() -risk_free_return) / downside_volitality
annualized_sortino_ratio = ((log_returns.mean() - risk_free_return) / downside_volitality.std()) * np.sqrt(TOTAL_DAYS)
sortino_ratio = sortino_ratio.dropna()

fig = sortino_ratio.plot.line(title=f"Sortino Ratio of portfolio assets with {TRADING_DAYS} as period", labels={
    "index": "Days",
    "value": "Sortino Ratio",
    "variable": "Legend"
})
st.plotly_chart(fig, use_container_width=True)
st.markdown("#### Annualized Sortino ratio for the stocks in the portfolio")
for i in range(len(annualized_sharpe_ratio)):
    st.write(
        f"{stocks[i]} has a annualized sortino ratio of: {annualized_sortino_ratio[i]}")

st.title("")
st.markdown("### M2 Ratio for various portfolio assets")
BENCHMARK = stocks[-1]
benchmark_volitality = volitality[BENCHMARK]
m2_ratios = pd.DataFrame()
for i in log_returns.columns:
    if i != BENCHMARK:
        m2_ratio = sharpe_ratio[i] * benchmark_volitality + risk_free_return
        m2_ratios[i] = m2_ratio
fig = m2_ratios.plot.line(title=f"M2 Ratio of portfolio assets with {TRADING_DAYS} as period", labels={
    "index": "Days",
    "value": "M2 Ratio",
    "variable": "Legend"
})
st.plotly_chart(fig, use_container_width=True)
