import os
import configparser

from flask import Flask


app = Flask(__name__, instance_relative_config=True)
config = configparser.ConfigParser()
config.read('dev.cfg')

app.config.from_mapping(
    HOST=config.get('CLUSTER', 'HOST'),
    DB_NAME=config.get('CLUSTER', 'DB_NAME'),
    DB_USER=config.get('CLUSTER', 'DB_USER'),
    DB_PASSWORD=config.get('CLUSTER', 'DB_PASSWORD'),
    DB_PORT=config.get('CLUSTER', 'DB_PORT')
)

from visual import db
db.init_app(app)

from visual import graphic
app.register_blueprint(graphic.bp)
app.add_url_rule('/', endpoint='index')
