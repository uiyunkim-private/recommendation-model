import os

import boto3


def get_session():
    return boto3.Session(
        aws_access_key_id='AKIAXXPRRAWGIVPKHXSN',#os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key='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',#=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name= 'ap-northeast-2'#os.getenv('AWS_DEFAULT_REGION')
    )


def build_s3():
    session = get_session()
    s3 = session.resource('s3')
    return s3
