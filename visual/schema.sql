DROP TABLE IF EXISTS cryptos;

-- id could be a BIGSERIAL in Postgres, and wouldn't need AUTOINCREMENT

CREATE TABLE cryptos (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "crypto" VARCHAR(12) NOT NULL,
    "date" DATE NOT NULL,
    "close_price" REAL NOT NULL
);