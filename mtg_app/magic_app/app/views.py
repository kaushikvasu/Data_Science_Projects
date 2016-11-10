import logging
import json

from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
import pandas as pd
import numpy as np
from magicModeling import predict

from . import app


logger = logging.getLogger('app')

class PredictForm(Form):

    magic_card = fields.StringField('Select a Magic Card (Through Kaladesh):', validators=[Required()])
  
    submit = fields.SubmitField('Submit')


@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    prediction = None
    prediction_table = None
    try:
        form.validate_on_submit()
        # store the submitted values
        submitted_data = form.data

        # Retrieve values from form
        magic_card = str(submitted_data['magic_card'])
        prediction = predict(magic_card)
        prediction_table = pd.DataFrame(prediction).to_html().replace('border="1"','border="0"')

        #if prediction

    except:
        print(form.errors)

    return render_template('index.html',
                            form=form,
                            prediction_table=prediction_table,
                            prediction=prediction)