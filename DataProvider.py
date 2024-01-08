from datetime import timedelta
import pandas as pd
import os


class DataProvider:
    def __init__(self, start_date='2020-01-01', end_date='2024-01-01'):
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self):
        cryptos = pd.DataFrame()
        for filename in os.listdir("data-master/csv"):
            if 'csv' in filename:
                crypto = pd.read_csv("data-master/csv" + '/' + filename, encoding='Windows-1252')
                if 'Date' in crypto.columns:
                    crypto['Date'] = pd.to_datetime(crypto['Date']).dt.date
                    crypto = crypto[(crypto['Date'] >= self.start_date) & (crypto['Date'] <= self.end_date)].reset_index(drop=True)
                    if len(crypto['Date']) > 0 and crypto['Date'].iloc[0] == self.start_date:
                        if not 'Date' in cryptos.columns:
                            cryptos['Date'] = crypto['Date']
                        cryptos[filename.split('.')[0]] = crypto['Open'].reset_index(drop=True)

            cryptos = cryptos.dropna(axis='columns')
        cryptos = cryptos.set_index('Date', drop=True)

        return cryptos

    def get_data_for_dates(self, data, start_date='2020-01-01', end_date='2024-01-01'):
        data = data[(data.index > start_date) & (data.index < end_date)]

        return data

    def get_dates(self, increment):
        current = self.start_date
        list_dates = [self.start_date]

        while current + timedelta(days=increment) < self.end_date:
            list_dates.append(current + timedelta(days=increment))
            current = current + timedelta(days=increment)

        return list_dates
