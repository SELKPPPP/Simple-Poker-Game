import urllib.request
import json

def lambda_handler(event, context):
    url = "http://50.18.67.156:5001/model/train"
    req = urllib.request.Request(url, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            return {
                'statusCode': 200,
                'body': json.loads(response.read())
            }
    except Exception as e:
        return {'statusCode': 500, 'body': str(e)}