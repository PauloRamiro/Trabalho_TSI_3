from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
from pandas import DataFrame
import pandas as pd
import streamlit as st
import data_pre_processing
import prediction


#st.set_page_config(page_tittle = "Prophet bot")

sidebar_data_list = []


class Header:
    @staticmethod
    def print_it():
        st.title("Analise das ações")


class Sidebar:
    @staticmethod
    def print_it():
        global sidebar_data_list
        ticker_name_select = Sidebar.select_ticker()
        selected_period = Sidebar.select_period()
        selected_real_comp = "Predição"
        selected_business  = Sidebar.select_business()

        sidebar_data_list = [ticker_name_select, selected_period, selected_real_comp, selected_business]

    @staticmethod
    def select_ticker():
        df_tickers = (data_pre_processing.DataTicker.collecting_data_name_in_csv())["ticker_company"]

        st.sidebar.header("Área de Ações")
        st.sidebar.write("Nessa seção você irá selecionar a ação a ser analisada na predição,"
                         + " veja/digite a ação, logo abaixo:")
        ticker_name_select = st.sidebar.selectbox("Escolha uma ação:", df_tickers)
        ticker_name_select = (ticker_name_select.split('-')[0] + ".SA")  # select ticker code
        return ticker_name_select

    @staticmethod
    def select_period():
        periods = ["1 dia"]

        st.sidebar.header("Área de Periodos")
        st.sidebar.write("Nessa seção você irá selecionar o periodo a ser analisado na predição,"
                         + " veja os periodos abaixo:")
        selected_period = st.sidebar.selectbox("Escolha um periodo:", periods)
        return selected_period
    
    def select_business():
        st.sidebar.header("Área de Simulações")

        st.sidebar.write("Nessa seção você irá selecionar o dia em que você comprou as ações da empresa escolhida "
                         + "e a quantidade de ações compradas.")
        
        date = st.sidebar.date_input("Dia de compra das ações:")
        amount_tickers = st.sidebar.number_input("Quantidade de ações:")
        return date, amount_tickers


class Body:

    @staticmethod
    def print_it():
        ticker_name_select = sidebar_data_list[0]  # Collected global variable data, extracted from the Sidebar class
        selected_period = sidebar_data_list[1]  # Collected global variable data, extracted from the Sidebar class
        selected_real_comp = sidebar_data_list[2]  # Collected global variable data, extracted from the Sidebar class

        df_ticker = data_pre_processing.DataTicker.collecting_data_in_yfinance(ticker_name_select, selected_period)

        Body.show_data_graph(selected_period, ticker_name_select, df_ticker)
        Body.show_data_prediction(selected_real_comp, df_ticker)

    @staticmethod
    def show_data_graph(selected_period, ticker_name_select, df_ticker):
        st.subheader("Tabelas de valores  -  " + ticker_name_select[:-3])
        st.write(df_ticker)

        st.download_button(
            "Baixar tabela como CSV",
            df_ticker.to_csv(index=False).encode('utf-8'),
            f"{ticker_name_select[:-3]} - {selected_period}.csv",
            "text/csv",
            key='download-csv'
            )

        st.subheader("Gráfico de preços de fechamentos ajustados")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ticker["Datetime"],
                                 y=df_ticker["Adj Close"],
                                 name="Fechamento ajustado",
                                 line_color="blue"))
        st.plotly_chart(fig)

    @staticmethod
    def show_data_prediction(selected_real_comp, df_ticker):
        selected_business_date = sidebar_data_list[3][0]
        st.subheader(selected_real_comp)
        number_periods_forecast = st.slider("Quantidade de periodos para previsão: ",
                                            5, 60)
        st.write("O modelo está sendo treinado e isso pode demorar um pouco, por favor aguarde...")

        model, predict = prediction.Training(df_ticker=df_ticker,
                                                 number_periods_forecast=number_periods_forecast).prophet_prediction()
        
        
        st.write(pd.to_date(df_ticker["Datetime"].iloc[-1]))
        st.write(selected_business_date)
        teste = df_ticker.loc[df_ticker["Datetime"]>= selected_business_date]
        st.write(teste)
        difference = df_ticker["Adj Close"].copy() - predict["yhat"].iloc[:-(number_periods_forecast)].copy()
        average_difference = difference.iloc[-15:].mean()
        difference = DataFrame(difference.iloc[-15:].to_list() + [average_difference]*number_periods_forecast, columns=["difference"])

        corrected_predict = DataFrame(predict["yhat"].iloc[-(number_periods_forecast+15):].to_list(), columns =["yhat"])
        corrected_predict = corrected_predict["yhat"].copy() + difference["difference"].copy()

        predict = predict.iloc[-(number_periods_forecast+15):]

        #st.write(predict[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        st.subheader("Gráfico com os valores de predição:")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=predict["ds"],
                                 y=corrected_predict,
                                 name="Valores preditos",
                                 line_color="blue"))
        st.plotly_chart(fig2)


        #model_predict_graphic = plot_plotly(model, predict)
        #st.plotly_chart(model_predict_graphic)

        #st.subheader("Gráficos com dados técnicos da predição realizada:")
        #components_model_predict_graphic = plot_components_plotly(model, predict)
        #st.plotly_chart(components_model_predict_graphic)
