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
  "data": [[ 0.93885142, -1.01419247, -0.60813278, -0.45617155, -0.20533467,
        -0.79985861, -0.58655899, -0.5349335 ,  0.95782629, -0.75592895,
         0.75592895, -0.13050529, -0.21213203, -0.30229756, -0.17916128,
        -0.14834045, -0.13968606, -0.04897021,  0.52752958, -0.56814154,
         2.84375747, -1.35067551, -0.06933752, -0.04897021, -0.04897021,
        -0.22999288, -0.47896948,  0.86120071, -0.45617155, -0.04897021,
        -0.06933752,  0.80757285, -0.15655607, -0.15655607, -0.72879046,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ]]
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
