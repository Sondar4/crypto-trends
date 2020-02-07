import csv, sqlite3

import click
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

# Function to initialize the database from scratch
def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))
    
    #Load csv into database
    with open('C:/Users/Ramon/Scripts/python/x-visual/cryptos-py.csv','r') as fin:
        # csv.DictReader uses first line in file for column headings by default
        names = ['bitcoin-cash', 'bitcoin-sv', 'bitcoin', 'ethereum', 'ripple']
        dr = csv.DictReader(fin) # comma is default delimiter
        to_db = [(i['slug'], i['date'], i['close']) for i in dr if i['slug'] in names]

    db.executemany('INSERT INTO cryptos (crypto, date, close_price) VALUES (?, ?, ?);', to_db)
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