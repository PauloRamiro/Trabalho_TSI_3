from prophet import Prophet
from sklearn.model_selection import train_test_split


class Training:
    def __init__(self, df_ticker, number_periods_forecast=30):
        self.df_ticker = df_ticker
        self.number_periods_forecast = number_periods_forecast

    def prophet_prediction(self):
        self.df_ticker = self.df_ticker[["Datetime", "Adj Close"]]
        self.df_ticker = self.df_ticker.rename(columns={"Datetime": "ds", "Adj Close": "y"})
        model = Prophet(seasonality_mode="multiplicative")
        model.fit(self.df_ticker)
        future = model.make_future_dataframe(periods=self.number_periods_forecast,
                                             freq='B')
        predict = model.predict(future)
        return model, predict

    def prophet_comparison(self):
        self.df_ticker = self.df_ticker[["Datetime", "Adj Close"]]
        self.df_ticker = self.df_ticker.rename(columns={"Datetime": "ds", "Adj Close": "y"})
        df_train, df_test = train_test_split(self.df_ticker, test_size=0.05,
                                             shuffle=False)

        model = Prophet(seasonality_mode="multiplicative")
        model.fit(df_train)
        predict = model.predict(df_test)
        return model, predict

