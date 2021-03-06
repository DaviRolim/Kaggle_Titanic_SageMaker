import os
import io
import boto3
import json
import csv

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    
    print("data: " + json.dumps(event))
    '''{
  		"data": "1.       ,  0.       ,  0.       ,  0.       ,  1.       ,         0.       ,  0.       ,  1.       ,  0.       ,  0.       ,         0.       ,  1.       ,  0.       ,  0.       ,  0.       ,         1.       ,  0.       , -0.5031762"
	   }
    '''
    print(type(event))
	
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=event['data'])

    result = json.loads(response['Body'].read().decode())

    print(result)

    return result
