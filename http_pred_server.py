import sys
import os

import pickle as pickle
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from flask import Flask, jsonify, request

app = Flask(__name__)

def float_to_str(f):
    """
    Stores floats as string in a way that works with scientific notation.

    Otherwise when unpacking a json, may be hard to import into for example
    AWS Redshift.
    """
    float_string = repr(f)
    if 'e' in float_string:  # detect scientific notation
        digits, exp = float_string.split('e')
        digits = digits.replace('.', '').replace('-', '')
        exp = int(exp)
        zero_padding = '0' * (abs(int(exp)) - 1)  # minus 1 for decimal point in the sci notation
        sign = '-' if f < 0 else ''
        if exp > 0:
            float_string = '{}{}{}.0'.format(sign, digits, zero_padding)
        else:
            float_string = '{}0.{}{}'.format(sign, zero_padding, digits)
    return float_string


@app.route('/predict', methods=['Post'])
def prediction_server_API_call():
    # first read the json and turn it info a DataFrame
    try:
        data_json = request.get_json(force=True)
        d = pd.read_json(json.dumps(data_json), orient='records')
    except Exception as e:
        raise e

    if d.empty:
        return(bad_request())
    else:

        # Now get the current directory to load the feature list, transformer
        # and serialized model:
        current_dir = os.path.realpath(
                            os.path.join(
                                os.getcwd(),
                                os.path.dirname(__file__)))

        # Now load the feature to use for the model:
        features = pickle.load(open(os.path.join(
                                        str(current_dir),
                                        "latest_features.fl"), "rb"))

        # Now load the sklearn-pandas transformer pipeline:
        transformer = pickle.load(open(os.path.join(
                                            str(current_dir),
                                            "latest_transformer.tr"), "rb"))

        # and finally load the serialized xgboost model itself:
        model = pickle.load(open(os.path.join(
                                    str(current_dir),
                                    "latest_model.mdl"), "rb"))

        # save the identifiers so that we can add them in the return json:
        ids = d.id

        # Only select the features that we use in our model:
        d = d[features]

        # run the model through our transformer pipeline (filling missing values,
        # fixing categorical variables, etc):
        d = transformer.transform(d)

        # Turn the DataFrame in an xgboost compatible format:
        X = xgb.DMatrix(d)

        # and run the prediction:
        model_prediction = model.predict(X)

        # Turn the prediction in a json friendly string format:
        # (taking care of scientific notation for example:)
        prediction_str = pd.Series(np.round(prediction_round,3))\
                            .apply(float_to_str)

        # turn the prediction into a DataFrame of with id and prediction:
        prediction_df = pd.DataFrame(
                            {
                                'id': d.id,
                                'prediction' : prediction_str
                             })

        # turn into a json
        return_json = prediction_dataframe.to_json(orient='records')

        # return the json
        return return_json


if __name__ == "__main__":
    app.run()
