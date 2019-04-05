#### Guide to Deploying Machine Learning algorithms to AWS Lambda using Zappa

Last updated: January 22, 2019

By: Oege Dijk

## Purpose:
AWS lambda is a very nice serverless implementation that can be used to serve
machine learning models. You can train your model somewhere else
(e.g. on an EC2 instance), then build a Flask server that takes in data,
processes the data, runs your model on the data, and returns the prediction.

The nice thing about AWS Lambda is that the server only runs when it gets a
request, and you only pay by the miliseconds while the request is running.
It also scales automatically (every request spins up a new server instance),
and is easy on maintenance (that is, zero maintenance).

To get your server running on Lambda there is a very nice package called
Zappa (https://github.com/Miserlou/Zappa/). This package does most of the
difficult part of setting up the lambda server for you as long as your
server is built on Python and either on top of Flask or Django.

However it is mainly built for deploying web applications, so it requires
some tweaking to get it to work with serving machine learning predictions.
Hence this guide.

## REST prediction
This is the server code that I was trying to get to work in a file
called `http_pred_server.py`. It uses Flask to serve machine learning
predictions in a REST framework. It reads in data through a JSON POST,
loads the machine learning model, and some files that indicate which
columns to use for preprocessing,
preprocesses and transforms the data, then runs the prediction model
and returns the prediction as again as a JSON.

```

import sys
import os

import pickle as pickle
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/predict', methods=['Post'])
def prediction_server_API_call():
    try:
        data_json = request.get_json(force=True)
        d = pd.read_json(json.dumps(data_json), orient='records')
    except Exception as e:
        raise e

    if d.empty:
        return(bad_request())
    else:

        current_dir = os.path.realpath(os.path.join(os.getcwd(),
                                os.path.dirname(__file__)))
        features = pickle.load(open(os.path.join(str(current_dir),
                                        "latest_features.fl"), "rb"))
        transformer = pickle.load(open(os.path.join(str(current_dir),
                                            "latest_transformer.tr"), "rb"))
        model = pickle.load(open(os.path.join(str(current_dir),
                                    "latest_model.mdl"), "rb"))

        ids = d.id
        d = d[features]
        d = transformer.transform(d)
        X = xgb.DMatrix(d)

        model_prediction = model.predict(X)
        prediction_str = pd.Series(np.round(prediction_round,3))\
                            .apply(float_to_str)

        prediction_df = pd.DataFrame(
                            {
                                'id': d.id,
                                'prediction' : prediction_str
                             })
        return prediction_dataframe.to_json(orient='records')
```

Now, this is easy enough to get to work locally, however how do we get it
to work on AWS Lambda?


## Step 1: Setup environment:

First step is to make deployment directory and set up a virtual environment.
*Always* first set up the environment and *only then* install zappa,
otherwise it won't work. Zappa will basically copy, compile and package your
entire virtual env over to Lambda, so always do this first:

```
mkdir lambda-deployment
cd lambda-deployment
python3 -m venv zap-env
source zap-env/bin/activate
```


Now move your serialized model, your data transformer classes, your
serialized feature lists, the code that gets called by your transformer
classes, and whatever else that your code needs to run to the
`lambda-deployment/` directory, and make sure that the `requirements.txt` file
is up-to-date. (I personally use the `pipreqs` package).

The install all the dependencies:

```
pip install -U pip setuptools
pip install -r requirements.txt
```

### Non Python libraries

When using python packages that are dependent on C libraries
(such as `xgboost` in our case), you need to compile these on a proper
linux environment compatible with the linux setup of Lambda (I did on an
  EC2 server), and then copy the `.so` file to a `/lib` subdirectory in
  your project directory. To compile from source:

```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4; cd ..
```

(you may have to run `sudo apt-get install build-essential` first
  to install the c compiler)

Alternatively you can compile the `libxgboost.so` library inside a Docker
container resembles the actual lambda environment with the following Dockerfile:

```
FROM lambci/lambda:build

RUN git clone --recursive https://github.com/dmlc/xgboost
RUN cd xgboost; make -j4
```

Then you can copy the `libxgboost.so` from the Docker image:

```
docker cp lambda_xgboost:/var/task/xgboost/lib/libxgboost.so .
```


### Installing zappa:

Now we install `zappa`:

`pip install zappa`

In order for zappa to do its magic, it needs your AWS credentials. Make sure
the sysadmins give you the right credentials that you need to deploy,
update and undeploy the lambda server. The credentials themselves need to
be stored in `~/.aws/credentials` like so:

```
[default]
aws_access_key_id=[your_key_goes_here]
aws_secret_access_key=[your_secret_key_goes_here]
```

Also, choose your default region in `~/.aws/config`

```
[default]
region=eu-central-1
```

## Step 2: Intialize Zappa:

Now we can initialize zappa:

`zappa init`

This walks you through a couple of questions to set up your lambda server.
The name of your app function is the name of your python file with the
Flask/Django code excluding the `.py`, followed by a . and the name of your
Flask/Django app. So in our case this is `http_pred_server.app`.

Depending on the size of your project and whether you are using python
packages with C libraries, you need to first edit the zappa_settings.json file.
AWS Lambda has a size limit of 50MB (easy to reach if you have a couple
  of dependencies). However if you store some of the code in S3 and only
  load it as necessary you can cut down on the filesize. Zappa does this for
  you if you include `"slim_handles":"true"`. You may also want to add
  `"keep_warm":false`, as long your application can wait for a few seconds
  to start up cold.

In order to help lambda find your compiled C libraries, you need to tell
zappa where they live by specifying e.g. `"include": ["lib/libxgboost.so"]`

Also make sure there is enough memory to run the application (max 3008MB),
and set a timeout long enough for the predictions to run (max 900 seconds):

`zappa_settings.json`:

```
{
    "production": {
        "aws_region": "eu-central-1",
        "profile_name": "default",
	      "keep_warm":"false",
        "project_name": "zap-server",
        "runtime": "python3.6",
        "s3_bucket": "YOUR_ZAPPA_DEPLOYMENT_S3_BUCKET",
        "app_function": "http_pred_server.app",
        "memory_size": 3008,
        "timeout_seconds": 900,
	      "slim_handler":"true",
	      "include": ["lib/libxgboost.so"],
        "events": [{
            "function": "s3_pred_server.run_prediction",
            "event_source": {
                  "arn":  "arn:aws:s3:::YOUR_UPLOAD_BUCKET",
                  "events": [
                    "s3:ObjectCreated:*"
                  ]
               }
            }],
    }
}
```

## Step 3: Responding to S3 object creation events:

Serving your predictions through a flask server has a downside: AWS Lambda
only allows http payload up to about 4.5MB (officially 6MB, in practice it
  seems to crash around 4.5MB).
Luckily you can also configure your lambda function to respond to the
creation of for example csv files in a certain S3 bucket. This allows for
bigger file sizes, so that you can process more predictions at a time.

The code for this is located in `s3_pred_server.py`:

```
def run_prediction(event, context):
    # connect to s3:
    s3_resource = boto3.resource('s3')

    # Get the uploaded file's information
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    tmp_data_file_name = '/tmp/data_' + str(uuid.uuid4()) + '.csv'

    s3_resource.Object(bucket, key).download_file(tmp_data_file_name)

    #### RUN YOUR PREDICTION CODE HERE ####

    #### THEN SAVE RESULT TO tmp_result_file_name ####

    s3_resource.Bucket('YOUR_DOWNLOAD_BUCKET').upload_file(
        Filename=tmp_result_file_name, Key=key)

    os.remove(tmp_data_file_name)
    os.remove(tmp_result_file_name)
```



Now it is time to move your the server code

## Step 5: Actually deploying to AWS lambda

Now you can simply run:

`zappa deploy production`

(`production` is the name of the deployment that we specified in the
  `zappa_settings.json` file. You can for example specifiy a
  `dev`, `test` and `production` stage)


Zappa will now start compiling your code and dependencies in a lambda
compatible way, store the whole thing in a zip file, store the zip file in S3,
and start the AWS Lambda server from there, and finally give you an API
Gateway URL where you can find your server, and set up the S3 object
creation trigger. 

You need at least the following permissions:
- ApiGateway
- Lambda
- CloudFormation

(although even then sometimes it can be tricky. For one deployment my sysadmin
at some point gave up and just gave all permissions 30 minutes in order to deploy)

Below some example output:

```
Calling update for stage dev..
Downloading and installing dependencies..
 - scipy==1.1.0: Using locally cached manylinux wheel
 - scikit-learn==0.20.0: Using locally cached manylinux wheel
 - pandas==0.23.4: Using locally cached manylinux wheel
 - numpy==1.15.2: Using locally cached manylinux wheel
 - sqlite==python36: Using precompiled lambda package
Packaging project as gzipped tarball.
Target directory /home/ubuntu/premium-traffic/src/zappa-server/handler_venv/lib/python3.6/site-packages/zappa already exists. Specify --upgrade to force replacement.
You are using pip version 10.0.1, however version 18.0 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
Downloading and installing dependencies..
 - sqlite==python36: Using precompiled lambda package
Packaging project as zip.
Uploading zappa-server-dev-1538140749.tar.gz (80.7MiB)..
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 84.6M/84.6M [00:00<00:00, 104MB/s]
Uploading handler_zappa-server-dev-1538140806.zip (10.6MiB)..
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 11.1M/11.1M [00:00<00:00, 86.1MB/s]
Updating Lambda function code..
Updating Lambda function configuration..
Uploading zappa-server-dev-template-1538140812.json (1.6KiB)..
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1.64K/1.64K [00:00<00:00, 43.0KB/s]
Deploying API Gateway..
Scheduling..
Unscheduled zappa-server-dev-zappa-keep-warm-handler.keep_warm_callback.
Scheduled zappa-server-dev-zappa-keep-warm-handler.keep_warm_callback with expression rate(4 minutes)!
Your updated Zappa deployment is live!: https://zpekgff2kxrl.execute-api.eu-central-1.amazonaws.com/production
```


The next time you want to update your project you can run:

`zappa update production`

To inspect the logs in case of errors you run:

`zappa tail production`

or for only logs of the last hour:

`zappa tail --since 1h`


Finally to undeploy the whole thing you run

`zappa undeploy production`


## Using the Lambda HTTP server:

To use the server simply post data to the server and record the response. The following code downloads a sample from Redshift (for all rockman_id's that start with 'aa%' in the last two hours), saves it to a json, posts it to our AWS lambda instance, gets the response, and displays it in a nice Confusion matrix:

```
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}


print("Downloading data for server test...")
d = download_sample_from_redshift()

data_json = d.to_json(orient='records')

resp = requests.post("https://zpekf2kxrl.execute-api.eu-central-1.amazonaws.com/dev/predict", \
                    data = json.dumps(data_json),\
                    headers = header)


print("server status code: ", resp.status_code)
result = pd.DataFrame(resp.json())
```

## Using the Lambda S3 trigger:

In order to get prediction using the S3 trigger, you simply upload a .csv file with predictions and wait for a file with the same filename to appear in the other bucket:


```

s3_client = boto3.client('s3',
                        aws_access_key_id=[your_key_goes_here],
                        aws_secret_access_key=[your_secret_key_goes_here])

print("uploading csv to S3...")
s3_client.upload_file(
    Filename=local_file_name, Bucket='UPLOAD_BUCKET_NAME',
    Key=remote_file_name)

waiter = s3_client.get_waiter('object_exists')

print("waiting for predictions file to be ready for download...")

waiter.wait(Bucket='eu-s3-helix-ml-download', Key=remote_file_name)

print("downloading results...")
s3_client.download_file(Bucket='DOWNLOAD_BUCKET_NAME', Key=remote_file_name,
                        Filename=local_result_file_name)

```


## Conclusion:

Serving your ML models using AWS Lambda is pretty awesome, and pretty doable once you get the hang of it...
