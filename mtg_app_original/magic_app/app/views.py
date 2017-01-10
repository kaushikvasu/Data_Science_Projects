import logging
import json

from flask import render_template, request, jsonify
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
import pandas as pd
import numpy as np
from magicModeling import predict

from . import app


logger = logging.getLogger('app')

CARDS_DF = pd.read_csv("app/data/Magic_Pandas_DF.csv")

class PredictForm(Form):
    magic_card = fields.SelectField('Select a Magic Card (Through Kaladesh):', 
        choices=[],
        validators=[Required()])
  
    submit = fields.SubmitField('Submit')


@app.route("/search")
def search():
<<<<<<< HEAD:mtg_app(original)/magic_app/app/views.py
    # print(request.args)
=======
>>>>>>> 584351d12c8f60ac431a121b5a62859cb62de0aa:mtg_app/magic_app/app/views.py
    query = request.args.get("q")
    if not query:
        return jsonify({"results": []})
    query = str(query)
<<<<<<< HEAD:mtg_app(original)/magic_app/app/views.py
    # print(query)
    mask = CARDS_DF["name"].str.contains(query, case=False)
    # print(mask.sum())
    items = CARDS_DF.loc[mask, "name"][:100].tolist()
    # print(items)
=======
    mask = CARDS_DF["name"].str.contains(query, case=False)
    items = CARDS_DF.loc[mask, "name"][:100].tolist()
>>>>>>> 584351d12c8f60ac431a121b5a62859cb62de0aa:mtg_app/magic_app/app/views.py

    name_list = [{"text": item, "id": item} for item in items]
    return jsonify({"results": name_list})

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
