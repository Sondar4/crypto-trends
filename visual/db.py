import psycopg2
import click
import yfinance as yf

from flask import current_app, g
from flask.cli import with_appcontext

def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(            
            host = current_app.config['HOST'],
            dbname = current_app.config['DB_NAME'],
            user = current_app.config['DB_USER'],
            password = current_app.config['DB_PASSWORD'],
            port = current_app.config['DB_PORT']
        )

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        cur = db.cursor()
        cur.execute(f.read().decode('utf8'))

    # Get data From yahoo finance
    yf_codes = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'LTC-USD']
    codes_dict = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum',
                  'XRP-USD': 'ripple', 'BCH-USD': 'bitcoin-cash',
                  'LTC-USD': 'litecoin'}
    for code in yf_codes:
        crypto = codes_dict[code]
        t = yf.Ticker(code)
        h = t.history(period='max')
        h['Date'] = h.index
        h['Date'] = h['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        vals = [(crypto, row.Date, row.Close) for (_, row) in h.iterrows()]
        cur = db.cursor()
        cur.executemany('INSERT INTO cryptos (crypto, dt, close_price) VALUES (%s, %s, %s);', vals)
        db.commit() 


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    app.teardown_appcontext(close_db) # tells Flask to call that function when cleaning up after returning the response
    app.cli.add_command(init_db_command)
