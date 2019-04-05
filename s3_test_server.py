import pandas as pd
import boto3
import uuid

BASE_DIR = os.getcwd()

local_file_name = f'{BASE_DIR}/s3_test.csv'
local_result_file_name = f'{BASE_DIR}/s3_result.csv'
remote_file_name=f's3_test_{str(uuid.uuid4())}.csv'

# connecting to S3
s3_client = boto3.client('s3',
                         aws_access_key_id=[your_access_key],
                         aws_secret_access_key=[your_secret_key])

# upload csv file to S3 upload bucket
s3_client.upload_file(Filename=local_file_name,
                      Bucket='YOUR_UPLOAD_BUCKET',
                      Key=remote_file_name)

# wait for file to show up in download bucket
waiter = s3_client.get_waiter('object_exists')
waiter.wait(Bucket='YOUR_DOWNLOAD_BUCKET', Key=remote_file_name)

# download file from download bucket
s3_client.download_file(Bucket='YOUR_DOWNLOAD_BUCKET',
                        Key=remote_file_name,
                        Filename=local_result_file_name)

d_result = pd.read_csv(local_result_file_name)
print(d_result.head())
