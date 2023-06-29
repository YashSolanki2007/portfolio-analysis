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
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile


pd.options.plotting.backend = "plotly"
figures = []


st.title('Portfolio Analysis')
st.markdown("Portfolio Analysis is a free tool that you can use to analyze and improve your portfolio.")

stocks_inp = st.text_input('Enter tickers of the Portfolio seperated by commas with the benchmark as the last asset', value='GOOG')
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
figures.append(fig)

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
    figures.append(fig)



st.markdown("### Box plot of log returns")
for i in log_returns.columns:
    fig = log_returns[i].plot.box(labels={"value": "", "variable": i},
                                  title=f'Box and whisker plot of: {i}')
    st.plotly_chart(fig, use_container_width=True)
    figures.append(fig)



st.markdown("### Correlation Matrix for the various portfolio assets")
correlation_matrix = portfolio_df.corr()
fig = plt.figure()
ax = sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix for portfolio")
st.pyplot(fig)
figures.append(fig)

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
figures.append(fig)


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
figures.append(fig)

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
figures.append(fig)

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
figures.append(fig)


st.markdown("### Efficient Frontier for the portfolio")
num_portfolios = int(st.text_input("Enter the number of portfolios to generate", value="1000"))

num_assets = len(stocks)
mean_returns = log_returns.mean()
covariance_matrix = log_returns.cov()


portfolio_weights = []
portfolio_returns = []
portfolio_volitality = []

returns_n = portfolio_df.pct_change()
covariance_matrix = log_returns.cov() * 252
for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights / np.sum(weights) # Ensure that the weights sum up to one
    portfolio_weights.append(weights)
    # Use portfolio expected return formula 
    mean_returns = log_returns.mean() * 252
    returns = np.sum(weights * mean_returns) 
    portfolio_returns.append(returns)
    portfolio_volitality.append(np.sqrt(
        np.dot(weights.T, np.dot(log_returns.cov() * 252, weights.T))))
    
data = {'Returns': portfolio_returns, 'Volatility': portfolio_volitality}
for counter, symbol in enumerate(portfolio_df.columns.tolist()):
    data[symbol+' weight'] = [w[counter] for w in portfolio_weights]

portfolios = pd.DataFrame(data)

min_volitality_portfolio = min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
max_sharpe_ratio_portfolio = portfolios.iloc[((portfolios['Returns']-risk_free_return)/portfolios['Volatility']).idxmax()]

portfolios_scatter = go.Scatter(x=portfolios['Volatility'] * 100, y=portfolios['Returns'] * 100, mode='markers', name='Portfolios along efficient frontier')
min_volitality_portfolio_scatter = go.Scatter(
    x=[min_volitality_portfolio[1] * 100], y=[min_volitality_portfolio[0] * 100], marker=dict(color='red', size=14, line=dict(width=3, color='black')), 
    name='Minimum Volitality Portfolio')

max_sharpe_ratio_portfolio_scatter = go.Scatter(
    x=[max_sharpe_ratio_portfolio[1] * 100], y=[max_sharpe_ratio_portfolio[0] * 100], marker=dict(color='green', size=14, line=dict(width=3, color='black')), 
    name='Maximum Sharpe Ratio Portfolio')

data = [portfolios_scatter, min_volitality_portfolio_scatter,
        max_sharpe_ratio_portfolio_scatter]
layout = go.Layout(
    title='Portfolio Optimisation with the Efficient Frontier',
    yaxis=dict(title='Annualised Return (%)'),
    xaxis=dict(title='Annualised Volatility (%)'),
    showlegend=True,
    legend=dict(
        x=0.75, y=0, traceorder='normal',
        bgcolor='#E2E2E2',
        bordercolor='black',
        borderwidth=2),
    width=800,
    height=600)
fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig, use_container_width=True)
figures.append(fig)


st.markdown("#### Information Regarding the efficient frontier")
st.markdown("##### All the data given below is supposed to be interpreted as percentages(%)")

st.markdown("##### Minimum Volitality Portfolio Information")
min_volitality_portfolio_df = pd.DataFrame(min_volitality_portfolio[0:len(min_volitality_portfolio)] * 100)
st.dataframe(min_volitality_portfolio_df)

max_sharpe_ratio_portfolio_df = pd.DataFrame(max_sharpe_ratio_portfolio[0:len(max_sharpe_ratio_portfolio)] * 100)
st.markdown("##### Maximum Sharpe Ratio Portfolio Information")
st.dataframe(max_sharpe_ratio_portfolio_df)


# Calculating the beta and the expected returns of the portfolio and the individual assets (with CAPM)
benchmark_returns = log_returns[BENCHMARK]
weights = [max_sharpe_ratio_portfolio[2:].to_list()]
portfolio_returns = np.dot(weights, log_returns.T)
portfolio_returns = np.reshape(portfolio_returns, (portfolio_returns.shape[1]))
summed_portfolio_returns = np.sum(portfolio_returns) * 100

# Max sharpe ratio portfolio beta
portfolio_beta = pd.DataFrame(np.cov(portfolio_returns, benchmark_returns) / np.var(benchmark_returns))


assets_beta = []
for stock in stocks:
    asset_returns = log_returns[stock]
    asset_beta = np.cov(asset_returns, benchmark_returns) / np.var(benchmark_returns)
    asset_beta = asset_beta[0][1]
    assets_beta.append(asset_beta)

portfolio_beta = pd.DataFrame(portfolio_beta)[0][1]
# Assuming risk-free rate is 0
portfolio_expected_returns = portfolio_beta * (max_sharpe_ratio_portfolio[0] * 100)

assets_expected_returns = []
for i in range(len(stocks)):
    asset_expected_return = assets_beta[i] * ((log_returns[stocks[i]].mean() * 252) * 100)
    assets_expected_returns.append(asset_expected_return)

all_betas = assets_beta
all_betas.append(portfolio_beta)
cols = [i for i in stocks]
cols.append("Portfolio")
betas = pd.DataFrame([all_betas], columns=cols)
betas = betas.T
st.markdown("### Beta of the portfolio and each of it's stocks")
st.markdown("#### Ignore the beta of the benchmark.")
st.dataframe(betas)

cumulative_expected_returns = assets_expected_returns
cumulative_expected_returns.append(portfolio_expected_returns)
capm_expected_returns_df = pd.DataFrame([cumulative_expected_returns], columns=cols)
capm_expected_returns_df = capm_expected_returns_df.T
st.markdown("### Expected annualized returns of the assets as well as the portfolio according to the CAPM")
st.dataframe(capm_expected_returns_df)

# PDF Download Functionality
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


export_as_pdf = st.button("Export Report as PDF")

# Download analytics report option 
FONT_FAMILY = "Arial"
WIDTH = 210
HEIGHT = 297
name = ""

if export_as_pdf:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=0.0)

    # Main Page
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.ln(40)
    pdf.multi_cell(w=0, h=8, txt=f"Portfolio Analysis report")


    # First introduction page 
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt="Introduction")
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=5, txt=f"This technical report analyses the portfolio consisting the assets evaluated over a period of {compute_period} days: ")
    pdf.ln(8)
    pdf.set_font(FONT_FAMILY, size=13)
    for index, stock in enumerate(stocks, 1):
        pdf.cell(0, txt=f"{index}. {stock}")
        pdf.ln(6)


    # Second page close prices
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Close Prices')
    pdf.set_font(FONT_FAMILY, size=35)
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        figures[0].write_image(tmpfile.name)
        name = tmpfile.name
    pdf.image(name, x=5, y=20, w=WIDTH-20)
    name = ""


    pdf.ln(115)
    pdf.set_font(FONT_FAMILY, size=13)


    # Log returns page 
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Log returns')
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=7, txt="The Log returns can be used to understand whether the returns of each of the stocks of the portfolio is gaussian or not and also helps us understand the shape of the returns")
    # Histogram of the log returns
    y = 50
    initial = 0
    for i in range(len(stocks)):
        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            ind = i + 1
            figures[ind].write_image(tmpfile.name)        
            name = tmpfile.name

        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            ind2 = len(stocks) + 1 + i
            figures[ind2].write_image(tmpfile.name)
            name2 = tmpfile.name

        # pdf.image(name, x=5, y=(50 * i) + 50, h=50)
        # name = ""
        temp = y
        y = (50 * i) + 50

        new_page = False
        p1 = True
        if temp >= 250 and temp % 250 == 0:
            new_page = True
            p1 = False
        else:
            new_page = False
        if new_page:
            initial = 0
            pdf.add_page()
            pdf.image(name, x=5, y=initial, h=50)
            pdf.image(name2, x=100, y=initial, h=50)
        else:
            if p1:
                initial += 50
                pdf.image(name, x=5, y=initial, h=50)
                pdf.image(name2, x=100, y=initial, h=50)
            else:
                pdf.image(name, x=100, y=initial, h=50)
                pdf.image(name2, x=5, y=initial, h=50)
                initial += 50

        name = ""



    # Correlation Matrix
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Correlation Matrix')
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=7, txt="The Correlation matrix tells us how the movement of one stock/asset in the portfolio affects the movement of another stock/asset in the portfolio.")
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        ind = 2 * len(stocks) + 1
        figures[ind].savefig(tmpfile.name)
        name = tmpfile.name
    pdf.image(name, x=5, y=50, w=WIDTH-20)
    name = ""

    # Volitality 
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Volitality')
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=7, txt="The Volitality of a stock is an important thing to look at as it gives investors an idea about the risk assosciated with the invesment.")
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        ind = 2 * len(stocks) + 1 + 1
        figures[ind].write_image(tmpfile.name)
        name = tmpfile.name
    pdf.image(name, x=5, y=50, w=WIDTH-20)
    name = ""

    # Sharpe Ratio 
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Sharpe Ratio')
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=7, txt='The Sharpe Ratio of a stock/asset tells us wheather the returns provided by an investment are worth the volitality. Investors consider a Sharpe ratio greater than 1 to be pretty good.')
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        ind = 2 * len(stocks) + 1 + 2
        figures[ind].write_image(tmpfile.name)
        name = tmpfile.name
    pdf.image(name, x=5, y=50, w=WIDTH-20)
    name = ""
    pdf.ln(140)
    pdf.set_font(FONT_FAMILY, size=20)
    pdf.multi_cell(w=0, h=7, txt='The Annualized Sharpe Ratios for the assets in the portfolio is: ')
    pdf.ln(13)
    pdf.set_font(FONT_FAMILY, size=13)
    for i in range(len(annualized_sharpe_ratio)):
        pdf.cell(0, txt=f"The Annualized Sharpe Ratio for: {stocks[i]} is {annualized_sharpe_ratio[i]}")
        pdf.ln(7)

    # Sortino Ratio
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Sortino Ratio')
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=7, txt='The Sortino Ratio of a stock/asset is very similar to the Sharpe Ratio except it only considers downside volitality and ignores any upside volitality.')
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        ind = 2 * len(stocks) + 1 + 3
        figures[ind].write_image(tmpfile.name)
        name = tmpfile.name
    pdf.image(name, x=5, y=50, w=WIDTH-20)
    name = ""
    pdf.ln(140)
    pdf.set_font(FONT_FAMILY, size=20)
    pdf.multi_cell(
        w=0, h=7, txt='The Annualized Sortino Ratios for the assets in the portfolio is: ')
    pdf.ln(13)
    pdf.set_font(FONT_FAMILY, size=13)
    for i in range(len(annualized_sortino_ratio)):
        pdf.cell(
            0, txt=f"The Annualized Sortino Ratio for: {stocks[i]} is {annualized_sortino_ratio[i]}")
        pdf.ln(7)

    # M2 Ratio 
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='M2 Ratio')
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=7, txt='The M2 ratio measures the returns of the portfolio, adjusted for the risk of the portfolio relative to that of some benchmark.')
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        ind = 2 * len(stocks) + 1 + 4
        figures[ind].write_image(tmpfile.name)
        name = tmpfile.name
    pdf.image(name, x=5, y=50, w=WIDTH-20)
    name = ""

    # Efficient-Frontier
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Efficient Frontier')
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, size=13)
    pdf.multi_cell(w=0, h=7, txt='The Efficient Frontier consists of several portfolios that are generated. The Optimal Risk to return portfolio is the one with the maximum Sharpe Ratio which is denoted by the green dot.')
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        ind = 2 * len(stocks) + 1 + 5
        figures[ind].write_image(tmpfile.name)
        name = tmpfile.name
    pdf.image(name, x=5, y=50, w=WIDTH-20)
    name = ""
    pdf.ln(150)

    pdf.set_font(FONT_FAMILY, size=23)
    pdf.cell(0, txt='Minimum Volitality Portfolio Information')
    pdf.ln(10)

    pdf.set_font(FONT_FAMILY, size=13)
    min_volitality_dict = min_volitality_portfolio_df.to_dict()
    min_volitality_dict = min_volitality_dict[int(
        list(min_volitality_dict.keys())[0])]
    dict_keys = list(min_volitality_dict.keys())
    dict_values = list(min_volitality_dict.values())
    for i in range(len(min_volitality_dict)):
        pdf.cell(w=0, h=10, txt=f"{dict_keys[i]}: \t \t \t {dict_values[i]}%", border=1)
        pdf.ln(10)


    pdf.ln(30)
    pdf.set_font(FONT_FAMILY, size=23)
    pdf.cell(0, txt='Max Sharpe Ratio Portfolio Information')
    pdf.ln(10)

    pdf.set_font(FONT_FAMILY, size=13)
    max_sharpe_ratio_dict = max_sharpe_ratio_portfolio_df.to_dict()
    max_sharpe_ratio_dict = max_sharpe_ratio_dict[
        int(list(max_sharpe_ratio_dict.keys())[0])]
    dict_keys = list(max_sharpe_ratio_dict.keys())
    dict_values = list(max_sharpe_ratio_dict.values())
    for i in range(len(max_sharpe_ratio_dict)):
        pdf.cell(w=0, h=10, txt=f"{dict_keys[i]}: \t \t \t {dict_values[i]}%", border=1)
        pdf.ln(10)

    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=40)
    pdf.cell(0, txt='Beta and CAPM')
    pdf.ln(20)
    pdf.set_font(FONT_FAMILY, size=23)
    pdf.cell(0, txt='Portfolio beta information')
    pdf.ln(10)
    betas_dict = betas.to_dict()
    betas_dict = betas_dict[
        int(list(betas_dict.keys())[0])]
    dict_keys = list(betas_dict.keys())
    dict_values = list(betas_dict.values())
    pdf.set_font(FONT_FAMILY, size=13)
    for i in range(len(betas_dict)):
        pdf.cell(
            w=0, h=10, txt=f"{dict_keys[i]}: \t \t \t {dict_values[i]}%", border=1)
        pdf.ln(10)
    
    pdf.ln(30)

    pdf.set_font(FONT_FAMILY, size=23)
    pdf.cell(0, txt='Portfolio percent returns with CAPM')
    pdf.ln(10)
    capm_dict = capm_expected_returns_df.to_dict()
    capm_dict = capm_dict[
        int(list(capm_dict.keys())[0])]
    dict_keys = list(capm_dict.keys())
    dict_values = list(capm_dict.values())
    pdf.set_font(FONT_FAMILY, size=13)
    for i in range(len(capm_dict)):
        pdf.cell(
            w=0, h=10, txt=f"{dict_keys[i]}: \t \t \t {dict_values[i]}%", border=1)
        pdf.ln(10)


    # Saving the pdf file do this in the end
    html = create_download_link(pdf.output(
        dest="S").encode("latin-1"), f"Portfolio Analysis")
    st.markdown(html, unsafe_allow_html=True)
    st.text("")
