import boto3
from botocore import UNSIGNED
from botocore.client import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
s3.download_file('mobiactv2', 'MobiAct_Dataset_v2.0.rar', 'data/MobiActV2/MobiAct_Dataset_v2.0.rar')
