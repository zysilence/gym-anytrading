from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--format', default='forex', choices=['forex', 'stock'], help="csv format")
    parser.add_argument('-s', '--source', required=True, help="Row forex text file path")
    parser.add_argument('-o', '--output', required=True, help="Output data csv file path")
    parser.add_argument('-p', '--period', required=True, help="Time period(minutes) of the data:\n"
                                                              "1T: 1 minute\n"
                                                              "1H：1 hour\n;"
                                                              "1D: 1 day\n"
                                                              "1W: 1 week\n"
                                                              "1M: 1 month\n"
                                                              "1A: 1 year")

    args = parser.parse_args()

    source = args.source
    output = args.output
    period = args.period
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', source))
    df['Raw_time'] = df['Day'].apply(str) + df['Time'].apply(str).apply(lambda x: x.zfill(6))
    df['Time'] = pd.to_datetime(df['Raw_time'], format='%Y%m%d%H%M%S')
    # df['Timestamp'] = df['Timestamp'].astype('int64') // 1e9
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df.index = df['Time'].tolist()
    # columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volume_(Currency)', 'Weighted_Price']
    columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_out = pd.DataFrame(columns=columns)
    df_out['Open'] = df['Open'].resample(period).first()
    df_out['High'] = df['High'].resample(period).max()
    df_out['Low'] = df['Low'].resample(period).min()
    df_out['Close'] = df['Close'].resample(period).last()
    df_out['Volume'] = df['Volume'].resample(period).sum()
    # df_out['Volume_(Currency)'] = df['Volume'].resample(period).sum()
    # df_out['Weighted_Price'] = df['Close'].resample(period).mean()
    # df_out['Time'] = df_out.index.view('int64') // 1e9
    df_out['Time'] = df_out.index
    # df_out['Time'] = df['Time']
    # df_out = df_out.fillna(method='ffill')
    df_out = df_out.dropna(subset=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

    output_path = os.path.join(os.path.dirname(__file__), 'data', output)
    df_out.to_csv(output_path, index=0)


