import sys
import os

import pickle as pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from flask import Flask, jsonify, request
import json

import uuid


import boto3

def float_to_str(f):
    """
    Deal with scientific notation floats.
    """
    float_string = repr(f)
    if 'e' in float_string:  # detect scientific notation
        digits, exp = float_string.split('e')
        digits = digits.replace('.', '').replace('-', '')
        exp = int(exp)
        # minus 1 for decimal point in the sci notation:
        zero_padding = '0' * (abs(int(exp)) - 1)
        sign = '-' if f < 0 else ''
        if exp > 0:
            float_string = '{}{}{}.0'.format(sign, digits, zero_padding)
        else:
            float_string = '{}0.{}{}'.format(sign, zero_padding, digits)
    return float_string


def run_prediction(event, context):
    """
    Process a file upload.
    """
    s3_resource = boto3.resource('s3')

    # Get the uploaded file's information:
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # if you export from AWS Redshift to S3, will create manifest files:
    if key[-8:] != 'manifest':
        tmp_data_file_name = '/tmp/data_' + str(uuid.uuid4()) + '.csv'

        # download from the S3 to the /tmp/ directory of the lambda
        # serverless server:
        s3_resource.Object(bucket, key).download_file(tmp_data_file_name)

        # Then read the file as a pandas DataFrame:
        d = pd.read_csv(tmp_data_file_name)

        # and remove the file:
        os.remove(tmp_data_file_name)

        # Get the current dir to load feature list, transformer and serialized
        # model:
        current_dir = os.path.realpath(
                        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # Now load the feature to use for the model:
        features = pickle.load(open(os.path.join(str(current_dir),
                            "latest_features.fl"), "rb"))

        # Now load the sklearn-pandas transformer pipeline:
        transformer = pickle.load(open(os.path.join(str(current_dir),
                            "latest_transformer.tr"), "rb"))

        # and finally load the serialized xgboost model itself:
        model = pickle.load(open(os.path.join(str(current_dir),
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

        # Save the prediction DF to a csv file
        tmp_result_file_name = '/tmp/result_' + str(uuid.uuid4())
        prediction_dataframe.to_csv(tmp_result_file_name)

        ### ... and upload the csv to another bucket S3 bucket.
        UPLOAD_BUCKET_NAME = 'YOUR BUCKET NAME HERE'
        
        s3_resource.Bucket(UPLOAD_BUCKET_NAME).upload_file(
        Filename=tmp_result_file_name, Key=key)

        os.remove(tmp_result_file_name)

if __name__ == "__main__":
    app.run()
