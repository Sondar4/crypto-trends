DROP TABLE IF EXISTS cryptos;

CREATE TABLE IF NOT EXISTS cryptos (
    crypto VARCHAR(12) NOT NULL,
    dt TIMESTAMP NOT NULL,
    close_price NUMERIC(10, 5) NOT NULL,
    PRIMARY KEY (crypto, dt)
);
