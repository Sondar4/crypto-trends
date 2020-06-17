DROP TABLE IF EXISTS cryptos;

CREATE TABLE IF NOT EXISTS cryptos (
    crypto VARCHAR(12) NOT NULL,
    dt DATE NOT NULL,
    close_price DECIMAL(10, 5) NOT NULL,
    PRIMARY KEY (crypto, dt)
);