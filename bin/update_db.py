import configparser
import psycopg2
import os.path
import yfinance as yf

CONFIG_PATH = 'dev.cfg'

CRYPTOS = [
    'BTC-USD',
    'ETH-USD',
    'XRP-USD',
    'BCH-USD',
    'LTC-USD'
]
NAMES = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'XRP-USD': 'ripple',
    'BCH-USD': 'bitcoin-cash',
    'LTC-USD': 'litecoin'
}

def main():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host = config.get('CLUSTER', 'HOST'),
            dbname = config.get('CLUSTER', 'DB_NAME'),
            user = config.get('CLUSTER', 'DB_USER'),
            password = config.get('CLUSTER', 'DB_PASSWORD'),
            port = config.get('CLUSTER', 'DB_PORT')
        )
        cur = conn.cursor()
    except Exception as e:
        print('There was an error connecting to the database:')
        print(e)
        exit()

    # Update database
    for crypto in CRYPTOS:
        # Get last date on db
        cur.execute(
            "SELECT max(dt) FROM cryptos " \
               f"WHERE crypto='{NAMES[crypto]}';"
        )
        last_date = cur.fetchone()[0]
        last_date = last_date.strftime('%Y-%m-%d')
        
        # Get data from yahoo finance
        t = yf.Ticker(crypto)
        h = t.history(period='max', start=last_date)
        h['Date'] = h.index
        h = h[h.Date > last_date]
        
        #Insert new data
        vals = [(NAMES[crypto], row.Date, row.Close) for (_, row) in h.iterrows()]
        cur.executemany('INSERT INTO cryptos (crypto, dt, close_price) VALUES (%s, %s, %s);', vals)
        conn.commit()


if __name__=='__main__':
    main()
