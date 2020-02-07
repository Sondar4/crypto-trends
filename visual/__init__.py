import os

from flask import Flask


app = Flask(__name__, instance_relative_config=True)

app.config.from_mapping(
    SECRET_KEY=b'g~|\x13c\xecv\xe9\xacD*\x04L)o\xf6',
    DATABASE=os.path.join(app.instance_path, 'cryptos.sqlite'),
)

try:
    os.makedirs(app.instance_path)
except OSError:
    pass

from visual import db
db.init_app(app)

from visual import graphic
app.register_blueprint(graphic.bp)
app.add_url_rule('/', endpoint='index')