from flask import (
    Blueprint, redirect, render_template, request, url_for, send_file
)

#--------------------- Imports to generate the plot ---------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

from datetime import datetime
from random import randint
from pandas.plotting import register_matplotlib_converters
from os import remove
from matplotlib.dates import num2date
from io import BytesIO

#The number of simulations run to generate the ci
#A value too high will make the script go slow
n_sim = 101
#------------------------------------------------------------------------


def get_form(form):
    start_date = form['pred_from']
    months = form['pred_to']
    min_ci = form['low_perc']
    max_ci = form['high_perc']
    return start_date, months, min_ci, max_ci


def get_args(args):
    months = args.get('start')
    start_date = args.get('month')
    min_ci = args.get('minci')
    max_ci = args.get('maxci')

    try:
        # We only chekc for type compatibility as they would be
        # converted again to str when passed as arguments in the
        # template
        datetime.strptime(start_date, '%Y-%m-%d')
        int(min_ci)
        int(max_ci)
        return start_date, months, min_ci, max_ci

    except:
        return None


bp = Blueprint('graphic', __name__)

@bp.route('/', methods=('GET', 'POST'))
@bp.route('/bitcoin', methods=('GET', 'POST'))
def bitcoin():
    if request.method == 'POST':
        form = get_form(request.form)
        return redirect(url_for('graphic.bitcoin', month=form[0], start=form[1], minci=form[2], maxci=form[3]))
    
    return render_template('cryptos/bitcoin.html', params=get_args(request.args))


@bp.route('/ethereum', methods=('GET', 'POST'))
def ethereum():
    if request.method == 'POST':
        form = get_form(request.form)
        return redirect(url_for('graphic.ethereum', month=form[0], start=form[1], minci=form[2], maxci=form[3]))
    
    return render_template('cryptos/ethereum.html', params=get_args(request.args))


@bp.route('/ripple', methods=('GET', 'POST'))
def ripple():
    if request.method == 'POST':
        form = get_form(request.form)
        return redirect(url_for('graphic.ripple', month=form[0], start=form[1], minci=form[2], maxci=form[3]))
    
    return render_template('cryptos/ripple.html', params=get_args(request.args))


@bp.route('/bitcoin-cash', methods=('GET', 'POST'))
def bitcoinCash():
    if request.method == 'POST':
        form = get_form(request.form)
        return redirect(url_for('graphic.bitcoinCash', month=form[0], start=form[1], minci=form[2], maxci=form[3]))
    
    return render_template('cryptos/bitcoin-cash.html', params=get_args(request.args))


@bp.route('/bitcoin-sv', methods=('GET', 'POST'))
def bitcoinSV():
    if request.method == 'POST':
        form = get_form(request.form)
        return redirect(url_for('graphic.bitcoinSV', month=form[0], start=form[1], minci=form[2], maxci=form[3]))
    
    return render_template('cryptos/bitcoin-sv.html', params=get_args(request.args))


@bp.route('/image_plot/<crypto>/<start_date>/<months>/<min_ci>/<max_ci>', methods=('GET',))
def plot_prediction(crypto, start_date=datetime(2018, 2, 1), months=3, min_ci=0, max_ci=90):
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
    """

    #Set parameters
    sns.set_style('whitegrid')
    register_matplotlib_converters()
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    months = int(months)
    ci = [int(min_ci), int(max_ci)]

    #Load data
    data_path='cryptos-py.csv'
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

    sns.lineplot(data.date, data.close, color='#2980b9')

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

    # From: https://stackoverflow.com/questions/20107414/passing-a-matplotlib-figure-to-html-flask
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')


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