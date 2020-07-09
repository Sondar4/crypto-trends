import sqlite3
import click
import yfinance as yf

from flask import current_app, g
from flask.cli import with_appcontext

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

    # Get data From yahoo finance
    yf_codes = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD']
    codes_dict = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum',
                  'XRP-USD': 'ripple', 'BCH-USD': 'bitcoin-cash'}
    for code in yf_codes:
        crypto = codes_dict[code]
        t = yf.Ticker(code)
        h = t.history(period='max')
        h['Date'] = h.index
        h['Date'] = h['Date'].dt.strftime('%y-%M-%D %H:%M:%S')
        vals = [(crypto, row.Date, row.Close) for (_, row) in h.iterrows()]
        db.executemany('INSERT INTO cryptos (crypto, dt, close_price) VALUES (?, ?, ?);', vals)
        db.commit()

    #TODO: get data of btc-sv 


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    app.teardown_appcontext(close_db) # tells Flask to call that function when cleaning up after returning the response
    app.cli.add_command(init_db_command)
