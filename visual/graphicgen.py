import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from datetime import datetime
from random import randint
from pandas.plotting import register_matplotlib_converters
from os import remove

#The number of simulations run to generate the ci
#A value too high will make the script go slow
n_sim = 101

def GenerateFakeResults(original, results):
    """Generates fake results from fake data.
    Fake data is generated from mixing the resiudals of the fitted model.
    Then the model is trained on this fake data.
    
    original: data used to generate the original results
    results: a fitted least squares model
    
    returns: a fitted least squares model
    """
    fake = pd.DataFrame(original.num_date.copy())
    fake['close'] = results.fittedvalues + np.random.choice(results.resid, len(results.resid))
    formula = "close ~ num_date"
    model = smf.ols(formula, data=fake)
    return model.fit()


def PercentileLines(lines_seq, percents):
    """Returns a list with the percentile lines of the lines_seq.
    
    lines_seq: a list of NumPy arrays with the same shape.
    percents: a tuple or list of the desired percentiles.
    
    returns: a list of NumPy arrays.
    """
    lines_df = pd.DataFrame(lines_seq)
    quantiles = list() 
    
    for perc in percents:
        quantiles.append( pd.DataFrame(lines_df).quantile(q=perc/100, axis=0) )
        
    return quantiles


def plot_prediction(crypto, start_date=datetime(2018, 2, 1), months=3, ci=[0, 90], data_path='cryptos-py.csv'):
    """Generates a graph with the desired crypto values on 2017 and 2018.
    Then plots the intervals of confidence for future values using the least
    squares linear regression method.

    The dark grey zone is the ci for the sampling error.
    The light grey zone is the ci for the sampling error + random variation.

    The graph is saved in static/images/temp as (random_number).png.

    crypto: string, name of the crypto
    start_date: datetime object, the starting date for the regression
    months: int, the number of months we want to predict
    ci: list of len 2
    data_path: string

    returns: int, the number associated to the image generated
    """
    print(0)
    sns.set_style('white')
    register_matplotlib_converters()

    #Load data
    data = pd.read_csv('cryptos-py.csv', usecols=['slug', 'date', 'num_date', 'close'])
    data = data[data.slug == crypto]

    #Transform date strings to datetime objects
    data['date'] = data.date.apply(datetime.strptime, args=('%Y-%m-%d',)) #This makes the code go slow

    #Make first model
    formula = "close ~ num_date"
    model = smf.ols(formula, data=data[data.date > start_date])
    results = model.fit()

    #Create range of the days for the plot
    dates = pd.DataFrame()
    dates['num_date'] = pd.Series(range(months*30 + 365*2))
    dates['num_date'] += data.num_date.min()

    results_seq = []
    predicts_seq = []
    predicts_seq_with_res = []

    #Generate n_sim results with mixed data
    for i in range(n_sim):
        results_seq.append(GenerateFakeResults(data[data.date > start_date], results))
    
    #Generate predictions from fake results
    for i in range(n_sim):
        prediction = results_seq[i].predict(dates.num_date)
        predicts_seq.append(prediction)
        
    #Generate predictions from fake results and add mixed residuals
    for i in range(n_sim):
        prediction = results_seq[i].predict(dates.num_date)
        prediction += np.random.choice(results_seq[i].resid, len(prediction))
        predicts_seq_with_res.append(prediction)

    plt.figure(figsize=(14, 10))

    sns.lineplot(x='date', y='close', data=data, label='Real', color='#2980b9')

    quantiles = PercentileLines(predicts_seq, ci)
    quantiles_with_resid = PercentileLines(predicts_seq_with_res, ci)

    plt.fill_between(dates.num_date, quantiles[0], quantiles[1], alpha=0.5, color='#95a5a6')
    plt.fill_between(dates.num_date, quantiles_with_resid[0], quantiles_with_resid[1], alpha=0.3, color='#bdc3c7')

    
    plt.axvline(start_date, 0, 1, color='#34495e')

    #Generate fig name
    fig_code = randint(0, 999999999)
    fig_name = '{:09d}'.format(fig_code) + '.png'

    plt.savefig('static/images/temp/' + fig_name)
    return fig_code


def delete_image(fig_code):
    """Deletes the file with the associated code from the
    static/images/temp folder.

    code: int

    returns: None
    """
    fig_name = '{:09d}'.format(fig_code) + '.png'
    fig_path = 'static/images/temp/' + fig_name
    remove(fig_path)

    return None