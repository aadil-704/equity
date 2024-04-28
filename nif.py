import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from stocknews import StockNews
from prophet import Prophet  # Import Prophet here
import datetime


# Fetch fundamental data
def get_fundamental_data(symbol, period, statement):
    if period == 'annual':
        if statement == 'balance sheet':
            return yf.Ticker(symbol).balance_sheet.T
        elif statement == 'income statement':
            return yf.Ticker(symbol).financials.T
        elif statement == 'cash flow':
            return yf.Ticker(symbol).cashflow.T
    elif period == 'quarterly':
        if statement == 'balance sheet':
            return yf.Ticker(symbol).quarterly_balance_sheet.T
        elif statement == 'income statement':
            return yf.Ticker(symbol).quarterly_financials.T
        elif statement == 'cash flow':
            return yf.Ticker(symbol).quarterly_cashflow.T
    else:
        st.error('Wrong entry')

# Calculate moving averages
def calculate_moving_averages(data, short_window, long_window):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

# Generate buy/sell signals based on moving average crossovers
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['Buy'] = np.where(data['Short_MA'] > data['Long_MA'], 1.0, 0.0)
    signals['Sell'] = np.where(data['Short_MA'] < data['Long_MA'], -1.0, 0.0)
    signals['Signal'] = signals['Buy'] + signals['Sell']
    return signals

# Predict stock using Prophet
def predict_stock(symbol, data):
    data = data.reset_index()
    # Rename columns to 'ds' and 'y' as required by Prophet
    data = data.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Initialize Prophet model
    model = Prophet()
    
    # Fit the model
    model.fit(data)
    
    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=365)  # Predict for the next year
    
    # Predict
    forecast = model.predict(future)
    
    return model, forecast

def main():
    # Streamlit UI
    st.markdown(
        """
        <h1 style='text-align: center; color: black;'>📈Equity Research Screener📈</h1>
        """,
        unsafe_allow_html=True
    )

    # Sidebar inputs
    symbol = st.sidebar.text_input("Enter stock symbol", "WIPRO.NS")
    start_date = st.sidebar.text_input("Enter start date (YYYY-MM-DD)", "2023-01-01")
    end_date = st.sidebar.text_input("Enter end date (YYYY-MM-DD)", datetime.date.today().strftime('%Y-%m-%d'))  # Use today's date as the default
    short_window = st.sidebar.slider("Short window", 1, 100, 22)
    long_window = st.sidebar.slider("Long window", 1, 200, 44)  # Adjusted the range and default value
    period = st.sidebar.selectbox('Period', ['annual', 'quarterly'], key='period_selectbox')
    statement = st.sidebar.selectbox('Statement', ['balance sheet', 'income statement', 'cash flow'], key='statement_selectbox')
    selected_tab = st.sidebar.radio("Navigation", ["Company Profile", "Summary and Statistical Data", "Candlestick Chart", "Moving Averages and Signals", "Volume Data", "Fundamental Data", "Additional Information", "News", "Stock Prediction"])

    # Fetch data
    data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate moving averages
    data = calculate_moving_averages(data, short_window, long_window)

    # Generate signals
    signals = generate_signals(data)

    # Filter data based on start and end dates
    filtered_data = data.loc[start_date:end_date]

    # Fetch fundamental data
    fd_data = get_fundamental_data(symbol, period, statement)

    # Fetch additional company information
    ticker = yf.Ticker(symbol)

    if selected_tab == "Company Profile":
        # Display company profile
        st.subheader("Company Profile")
        info = ticker.info
        st.write("**Sector:**", info.get('sector', 'N/A'))
        st.write("**Industry:**", info.get('industry', 'N/A'))
        st.write("**Country:**", info.get('country', 'N/A'))
        st.write("**Website:**", info.get('website', 'N/A'))
        # Display key executives
        st.subheader("Key Executives")
        executives = info.get('companyOfficers', [])
        for executive in executives:
            st.write(executive['name'], "-", executive['title'])

    elif selected_tab == "Summary and Statistical Data":
        # Display summary of the stock and statistical data
        st.subheader("Summary and Statistical Data")
        try:
            info = ticker.info  # Moved the assignment here
            st.write("**Market Cap:**", info.get('marketCap', 'N/A'))
            st.write("**Forward PE Ratio:**", info.get('forwardPE', 'N/A'))
            st.write("**Trailing PE Ratio:**", info.get('trailingPE', 'N/A'))
            st.write("**Earnings Per Share (EPS):**", info.get('trailingEps', 'N/A'))  # Corrected EPS data

            st.write("**Dividend Yield:**", info.get('dividendYield', 'N/A'))
            st.write("**Beta:**", info.get('beta', 'N/A'))
            st.write("**Mean Close Price:**", filtered_data['Close'].mean())
            st.write("**Standard Deviation Close Price:**", filtered_data['Close'].std())
            st.write("**Minimum Close Price:**", filtered_data['Close'].min())
            st.write("**Maximum Close Price:**", filtered_data['Close'].max())
            if 'fiftyTwoWeekHigh' in info:
                st.write("**52-week High:**", info['fiftyTwoWeekHigh'])
            if 'fiftyTwoWeekLow' in info:
                st.write("**52-week Low:**", info['fiftyTwoWeekLow'])
        except Exception as e:
            st.error("Error: " + str(e))

    elif selected_tab == "Candlestick Chart":
        # Display candlestick chart
        st.subheader("Candlestick Chart")
        fig_candlestick = go.Figure()
        # Candlestick
        fig_candlestick.add_trace(go.Candlestick(x=filtered_data.index,
                                                  open=filtered_data['Open'],
                                                  high=filtered_data['High'],
                                                  low=filtered_data['Low'],
                                                  close=filtered_data['Close'], name='market data'))
        # Add 20-day SMA
        fig_candlestick.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Short_MA'], name='20-day SMA', line=dict(color='blue')))
        # Add 200-day SMA
        fig_candlestick.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Long_MA'], name='200-day SMA', line=dict(color='red')))
        # Update layout for candlestick chart
        fig_candlestick.update_layout(
            title='Candlestick chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(count=4, label="4h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="todate"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),  # Weekly time frame
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        st.plotly_chart(fig_candlestick)

    elif selected_tab == "Moving Averages and Signals":
        # Display moving averages and signals
        st.subheader("Moving Averages and Signals")
        fig = go.Figure()
        # Close price
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

        # Short moving average
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Short_MA'], mode='lines', name='Short Moving Average', line=dict(color='orange')))

        # Long moving average
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Long_MA'], mode='lines', name='Long Moving Average', line=dict(color='green')))

        # Buy signals
        buy_signals = filtered_data.loc[signals['Signal'] == 1.0]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Short_MA'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=8, color='green')))

        # Sell signals
        sell_signals = filtered_data.loc[signals['Signal'] == -1.0]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Short_MA'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=8, color='red')))

        # Update layout for the main figure
        fig.update_layout(
            title='Moving Averages and Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis=dict(tickformat='%Y-%m-%d'),
            yaxis=dict(type='linear'),
            showlegend=True
        )

        # X-Axes configuration for Plotly figure
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5m", step="minute", stepmode="backward"),
                    dict(count=4, label="4h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="todate"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),  # Weekly time frame
                    dict(step="all")
                ])
            )
        )

        # Show Plotly figure
        st.plotly_chart(fig)

    elif selected_tab == "Volume Data":
        # Display volume data
        st.subheader("Volume Data")
        volume_data = data['Volume']
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=volume_data.index, y=volume_data, name='Volume', marker=dict(color='blue')))
        fig_volume.update_layout(title="Volume", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_volume)

    elif selected_tab == "Fundamental Data":
        # Display fundamental data
        st.subheader("Fundamental Data")
        if fd_data is not None:
            st.write(fd_data.head())

            # Visualize fundamental data in a pie chart
            st.subheader("Fundamental Data Pie Chart:")
            labels = fd_data.columns
            values = fd_data.iloc[0].values
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values)])
            st.plotly_chart(fig_pie)

        else:
            st.error("Failed to retrieve fundamental data.")

    elif selected_tab == "Additional Information":
        # Display additional information
        st.subheader("Additional Information")
        st.write("**Major Holders:**")
        st.write(ticker.major_holders)

        st.write("**Insider Transactions:**")
        st.write(ticker.get_insider_transactions())

    elif selected_tab == "News":
        # News section
        st.subheader('Stock News')
        with st.expander("Latest News"):
            st.header(f'News of {symbol}')
            sn = StockNews(symbol, save_news=False)
            df_news = sn.read_rss()
            for i in range(10):
                st.subheader(f'News {i+1}')
                st.write(df_news['published'][i])
                st.write(df_news['title'][i])
                st.write(df_news['summary'][i])
                title_sentiment = df_news['sentiment_title'][i]
                st.write(f'Title Sentiment: {title_sentiment}')
                news_sentiment = df_news['sentiment_summary'][i]
                st.write(f'News Sentiment: {news_sentiment}')

    elif selected_tab == "Stock Prediction":
        # Stock prediction section
        st.subheader("Stock Prediction")
        st.markdown("""
        <style>
        .big-font {
            font-size:25px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">Note: For better performance in stock prediction, it is recommended to select a time period of more than 2 years.<br> Good luck!</p>', unsafe_allow_html=True)
        # Check if there is enough data for prediction
        if data is not None and not data.empty:
            # Train Prophet model and make predictions
            prophet_model, forecast_data = predict_stock(symbol, data)

            # Display Historical Raw Data
            st.subheader("Historical Raw Data")
            st.write(filtered_data)

            # Display Historical Raw Data Plot
            st.subheader("Historical Raw Data Plot")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=data.index, y=data['Open'], name="Historical Open", line=dict(color='blue')))
            fig_hist.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Historical Close", line=dict(color='green')))
            fig_hist.layout.update(title_text='Historical Raw Data of {}'.format(symbol), xaxis_rangeslider_visible=True)  # update layout
            st.plotly_chart(fig_hist)

            # Display Forecasted Raw Data
            st.subheader("Forecasted Raw Data")
            st.write(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

            # Display Forecasted Raw Data Plot
            st.subheader("Forecasted Raw Data Plot")
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines', name="Forecasted Close", line=dict(color='red')))
            fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_lower'], mode='lines', name="Forecasted Close Lower Bound", line=dict(color='rgba(255,0,0,0.3)')))
            fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_upper'], mode='lines', name="Forecasted Close Upper Bound", line=dict(color='rgba(255,0,0,0.3)')))
            fig_forecast.layout.update(title_text='Forecasted Raw Data of {}'.format(symbol), xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_forecast)

            # Display Forecast Plot
            st.subheader("Forecast Plot")
            fig = prophet_model.plot(forecast_data)
            st.plotly_chart(fig)

            # Display Forecast Components
            st.subheader("Forecast Components")
            fig_components = prophet_model.plot_components(forecast_data)
            st.write(fig_components)
        else:
            st.warning("Insufficient data for prediction. Please adjust the date range.")

if __name__ == "__main__":
    main()
