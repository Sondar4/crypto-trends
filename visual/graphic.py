from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

#from flaskr.auth import login_required
#from flaskr.db import get_db

#--------------------- Imports to generate the plot ---------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from datetime import datetime
from random import randint
from pandas.plotting import register_matplotlib_converters
from os import remove
from matplotlib.dates import num2date

#The number of simulations run to generate the ci
#A value too high will make the script go slow
n_sim = 101
#------------------------------------------------------------------------

bp = Blueprint('graphic', __name__)

@bp.route('/', methods=('GET', 'POST'))
@bp.route('/bitcoin', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        start_date = request.form['pred_from']
        months = request.form['pred_to']
        min_ci = request.form['low_perc']
        max_ci = request.form['high_perc']

        parameters = (start_date, months, [min_ci, max_ci])
        return redirect(url_for('index', month=months, start=start_date, minci=min_ci, maxci=max_ci))
    
    months = request.args.get('month')
    start_date = request.args.get('start')
    min_ci = request.args.get('minci')
    max_ci = request.args.get('maxci')

    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        months = int(months)
        min_ci = int(min_ci)
        max_ci = int(max_ci)
        params = (start_date, months, [min_ci, max_ci])
    except:
        params = (None,)

    #Create plot
    if all(params):
        fig_code = plot_prediction('bitcoin', *params)
        fig_path = '{:012d}'.format(fig_code) + '.png'
        return render_template('cryptos/bitcoin.html', image_url='/static/images/temp/' + fig_path)
    else:
        return render_template('cryptos/bitcoin.html', image_url='/static/images/plots/default_bitcoin.png')


@bp.route('/ethereum', methods=('GET', 'POST'))
def ethereum():
    if request.method == 'POST':
        start_date = request.form['pred_from']
        months = request.form['pred_to']
        min_ci = request.form['low_perc']
        max_ci = request.form['high_perc']

        parameters = (start_date, months, [min_ci, max_ci])
        return redirect(url_for('graphic.ethereum', month=months, start=start_date, minci=min_ci, maxci=max_ci))

    months = request.args.get('month')
    start_date = request.args.get('start')
    min_ci = request.args.get('minci')
    max_ci = request.args.get('maxci')

    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        months = int(months)
        min_ci = int(min_ci)
        max_ci = int(max_ci)
        params = (start_date, months, [min_ci, max_ci])
    except:
        params = (None,)

    #Create plot
    if all(params):
        fig_code = plot_prediction('ethereum', *params)
        fig_path = '{:012d}'.format(fig_code) + '.png'
        return render_template('cryptos/ethereum.html', image_url='/static/images/temp/' + fig_path)
    else:
        return render_template('cryptos/ethereum.html', image_url='/static/images/plots/default_ethereum.png')


@bp.route('/ripple', methods=('GET', 'POST'))
def ripple():
    if request.method == 'POST':
        start_date = request.form['pred_from']
        months = request.form['pred_to']
        min_ci = request.form['low_perc']
        max_ci = request.form['high_perc']

        parameters = (start_date, months, [min_ci, max_ci])
        return redirect(url_for('graphic.ripple', month=months, start=start_date, minci=min_ci, maxci=max_ci))

    months = request.args.get('month')
    start_date = request.args.get('start')
    min_ci = request.args.get('minci')
    max_ci = request.args.get('maxci')

    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        months = int(months)
        min_ci = int(min_ci)
        max_ci = int(max_ci)
        params = (start_date, months, [min_ci, max_ci])
    except:
        params = (None,)

    #Create plot
    if all(params):
        fig_code = plot_prediction('ripple', *params)
        fig_path = '{:012d}'.format(fig_code) + '.png'
        return render_template('cryptos/ripple.html', image_url='/static/images/temp/' + fig_path)
    else:
        return render_template('cryptos/ripple.html', image_url='/static/images/plots/ripple_default.png')


@bp.route('/bitcoin-cash', methods=('GET', 'POST'))
def bitcoinCash():
    if request.method == 'POST':
        start_date = request.form['pred_from']
        months = request.form['pred_to']
        min_ci = request.form['low_perc']
        max_ci = request.form['high_perc']

        parameters = (start_date, months, [min_ci, max_ci])
        return redirect(url_for('graphic.bitcoinCash', month=months, start=start_date, minci=min_ci, maxci=max_ci))

    months = request.args.get('month')
    start_date = request.args.get('start')
    min_ci = request.args.get('minci')
    max_ci = request.args.get('maxci')

    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        months = int(months)
        min_ci = int(min_ci)
        max_ci = int(max_ci)
        params = (start_date, months, [min_ci, max_ci])
    except:
        params = (None,)

    #Create plot
    if all(params):
        fig_code = plot_prediction('bitcoin-cash', *params)
        fig_path = '{:012d}'.format(fig_code) + '.png'
        return render_template('cryptos/bitcoin-cash.html', image_url='/static/images/temp/' + fig_path)
    else:
        return render_template('cryptos/bitcoin-cash.html', image_url='/static/images/plots/bcash_default.png')


@bp.route('/bitcoin-sv', methods=('GET', 'POST'))
def bitcoinSV():
    if request.method == 'POST':
        start_date = request.form['pred_from']
        months = request.form['pred_to']
        min_ci = request.form['low_perc']
        max_ci = request.form['high_perc']

        parameters = (start_date, months, [min_ci, max_ci])
        return redirect(url_for('graphic.bitcoinSV', month=months, start=start_date, minci=min_ci, maxci=max_ci))

    months = request.args.get('month')
    start_date = request.args.get('start')
    min_ci = request.args.get('minci')
    max_ci = request.args.get('maxci')

    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        months = int(months)
        min_ci = int(min_ci)
        max_ci = int(max_ci)
        params = (start_date, months, [min_ci, max_ci])
    except:
        params = (None,)

    #Create plot
    if all(params):
        fig_code = plot_prediction('bitcoin-sv', *params)
        fig_path = '{:012d}'.format(fig_code) + '.png'
        return render_template('cryptos/bitcoin-sv.html', image_url='/static/images/temp/' + fig_path)
    else:
        return render_template('cryptos/bitcoin-sv.html', image_url='/static/images/plots/bsv_default.png')


#--------------------- Functions to generate the plot ---------------------

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
    sns.set_style('whitegrid')
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

    plt.figure(figsize=(20, 14))

    sns.lineplot(x='date', y='close', data=data, color='#2980b9')

    quantiles = PercentileLines(predicts_seq, ci)
    quantiles_with_resid = PercentileLines(predicts_seq_with_res, ci)

    plt.fill_between(dates.num_date, quantiles[0], quantiles[1], alpha=0.5, color='#95a5a6')
    plt.fill_between(dates.num_date, quantiles_with_resid[0], quantiles_with_resid[1], alpha=0.3, color='#bdc3c7')

    # Make the plot more beatiful
    locs, labels = plt.yticks()
    for i in range(len(labels)):
        labels[i] = str(int(locs[i])) + ' $'
    plt.yticks(ticks=locs, labels=labels)
    plt.xticks(rotation=45)
    plt.tick_params(labelsize=20)
    plt.xlabel('', size = 30)
    plt.ylabel('', size = 30)
    
    plt.axvline(start_date, 0, 1, color='#34495e')

    x1,x2,y1,y2 = plt.axis()
    x1 = num2date(dates.num_date.min())
    x2 = num2date(dates.num_date.max())
    plt.axis((x1,x2,0,y2))

    #Generate fig name
    fig_code = randint(0, 999999999999)
    fig_name = '{:012d}'.format(fig_code) + '.png'

    plt.savefig('visual/static/images/temp/' + fig_name)
    return fig_code