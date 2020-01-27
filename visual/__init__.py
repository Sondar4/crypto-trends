from flask import Flask

app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY='dev',
)

from visual import graphic
app.register_blueprint(graphic.bp)
app.add_url_rule('/', endpoint='index')