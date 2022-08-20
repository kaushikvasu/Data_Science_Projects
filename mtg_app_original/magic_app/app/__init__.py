import logging
import joblib

from flask import Flask

app = Flask(__name__)
app.config.from_object("app.config")

from .views import *