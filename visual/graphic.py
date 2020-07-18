from flask import (
    Blueprint, redirect, render_template, request, url_for, send_file
)

from visual.db import get_db

#--------------------- Imports to generate the plot ---------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from random import randint
from pandas.plotting import register_matplotlib_converters
from os import remove
from matplotlib.dates import num2date
from io import BytesIO

#The number of simulations run to generate the ci.
#A value too high will make the app go slow.
N_SIMULATIONS = 100
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
        # template.
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


@bp.route('/litecoin', methods=('GET', 'POST'))
def litecoin():
    if request.method == 'POST':
        form = get_form(request.form)
        return redirect(url_for('graphic.litecoin', month=form[0], start=form[1], minci=form[2], maxci=form[3]))
    
    return render_template('cryptos/litecoin.html', params=get_args(request.args))


@bp.route('/image_plot/<crypto>/<start_date>/<months>/<min_ci>/<max_ci>', methods=('GET',))
def plot_prediction(crypto, start_date=datetime(2018, 2, 1), months=3, min_ci=0, max_ci=90):
    """Generates a graph of the values on 2017 and 2018 of the chosen crypto.
    Then plots the intervals of confidence for future values using the least
    squares linear regression method.

    The dark grey zone is the ci for the sampling error.
    The light grey zone is the ci for the sampling error + random variation.

    The graph is returned as an an URL with a PNG image.

    crypto: string, name of the crypto.
    start_date: datetime object, the starting date for the regression.
    months: int, the number of months we want to predict.
    min_ci: int, the lower percentile we want to plot.
    max_ci: int, the higher percentile we want to plot.
    """
    sns.set_style('whitegrid')
    register_matplotlib_converters()

    # Set the first date from which we make the prediction.
    start_date = datetime.strptime(start_date, '%Y-%m-%d')

    months = int(months)
    ci = [int(min_ci), int(max_ci)]

    db = get_db()
    cur = db.cursor()
    cur.execute(
        'SELECT dt, close_price'
        ' FROM cryptos'
        ' WHERE crypto = %s', (crypto,)
    )
    
    data = pd.DataFrame(cur.fetchall(), columns=['date', 'close_price'])

    data['id'] = data.index.values.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['close_price'] = pd.to_numeric(data['close_price'])

    # It's the same to make the regression on the id than on the date
    # converted to an int, so we make it on the id.
    # Otherwise we would have to transform dates into int type.
    formula = "close_price ~ id"
    model = smf.ols(formula, data=data[data.date > start_date])
    results = model.fit()

    first_date = data.date[0]                 # As the series is ordered,
    last_date = data.date[data.date.size-1]   # this is faster than using min and max.
    delta_days = days_between(last_date + relativedelta(months=months), first_date)
    time_df = pd.DataFrame(
        # This is more efficient than appending rows to a blank dataframe
        # because on each append pandas make a copy of the table.
        # With this method we avoid that.
        [{'id': i, 'date': first_date + timedelta(i)} for i in range(delta_days)]
    )

    fake_results = []
    fake_predictions = []
    fake_predictions2 = []

    for i in range(N_SIMULATIONS):
        fake_results.append(generate_fake_results(data[data.date > start_date], results, formula))

        prediction1 = fake_results[i].predict(time_df.id)
        fake_predictions.append(prediction1)

        # Add mixed residuals.
        prediction2 = prediction1.copy()
        prediction2 += np.random.choice(fake_results[i].resid, len(prediction2))
        fake_predictions2.append(prediction2)
    
    plt.figure(figsize=(20, 14))
    sns.lineplot(data.date, data.close_price, color='#2980b9')

    dark_quantiles = percentile_lines(fake_predictions, ci)
    light_quantiles = percentile_lines(fake_predictions2, ci)

    plt.fill_between(time_df.date, dark_quantiles[0], dark_quantiles[1], alpha=0.5, color='#95a5a6')
    plt.fill_between(time_df.date, light_quantiles[0], light_quantiles[1], alpha=0.3, color='#bdc3c7')

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

    x1,x2,_,y2 = plt.axis()
    x1 = time_df.date.min()
    x2 = time_df.date.max()
    plt.axis((x1,x2,0,y2))

    # From: https://stackoverflow.com/questions/20107414/passing-a-matplotlib-figure-to-html-flask
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')


#--------------------- Functions to generate the plot ---------------------
def days_between(d1, d2):
    """Returns the difference in days between two 
    datetime objects.
    """
    return abs((d2 - d1).days)


def generate_fake_results(original, results, formula):
    """Generates fake results from fake data.
    Fake data is generated from mixing the resiudals of the fitted model.
    Then the model is trained on this fake data.
    
    original: data used to generate the original results
    results: a fitted least squares model
    formula: a string with the formula
    
    returns: a fitted least squares model
    """
    fake = pd.DataFrame(original.id.copy()) 
    fake['close_price'] = results.fittedvalues + np.random.choice(results.resid, len(results.resid))
    model = smf.ols(formula, data=fake)
    return model.fit()


def percentile_lines(lines_seq, percents):
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
