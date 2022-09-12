
# Description: This program predicts the price of GOOG stock for a specific day
#              using the Machine Learning algorithm called Support Vector Regression (SVR)
#              & Linear Regression.

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import utils
import numpy


# Load the data
# from google.colab import files # Use to load data on Google Colab
# uploaded = files.upload() # Use to load data on Google Colab

def create_plot(dates, original_prices, ml_models_outputs):
    plt.scatter(dates, original_prices, color='black', label='Data')
    for model in ml_models_outputs.keys():
        plt.plot(dates, (ml_models_outputs[model])[0], color=numpy.random.rand(3, ), label=model)

    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Regression')
    plt.legend()
    plt.savefig("Plot.png")
    plt.show()


def train_predict_plot(file_name, df, ml_model):

    # print (df.head())

    ml_models_outputs = {}

    dates, prices, test_date, test_price = utils.getData(df)
    # utils.LSTM_model(dates, prices, test_date, df)
    for model in ml_model:
        method_to_call = getattr(utils, model)
        ml_models_outputs[model] = method_to_call(dates, prices, test_date, df)

    dates = list(df['date'])
    predict_date = dates[-1]
    dates = dates[:-3]
    # create_plot(dates, prices, ml_models_outputs)
    return dates, prices, ml_models_outputs, predict_date, test_price

# train_predict_plot('GOOG_30_days.csv', ['LSTM_model', 'elastic_net', 'BR'])

# print (all_files.keys())