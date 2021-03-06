import os
import io
import boto3
import json
import csv

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
'''
    event = {
  "data": [1.       ,  0.       ,  0.       ,  0.       ,  1.       ,         0.       ,  0.       ,  1.       ,  0.       ,  0.       ,         0.       ,  1.       ,  0.       ,  0.       ,  0.       ,         1.       ,  0.       , -0.5031762]
}
'''
    
    data = json.dumps(event['data'])
    print(data)

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=data.encode())

    result = json.loads(response['Body'].read().decode())

    print(result)

    return result
