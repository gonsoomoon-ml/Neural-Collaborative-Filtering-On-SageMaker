import json
import time
import boto3


def lambda_handler(event, context):
    """ 
    입력으로 세이지 메이커 모델, 앤드 포인트 컨피그 및 앤드 포인트 이름을 받아서, 앤드포인트를 생성 함.
    이미 존재하면 엔드포인트 업데이트 함.
    """
    
    # The name of the model created in the Pipeline CreateModelStep
    
    sm_client = boto3.client("sagemaker")

    ###################################
    # 입력 변수 저장
    ###################################
    
    model_name = event["model_name"]
    endpoint_config_name = event["endpoint_config_name"]
    endpoint_name = event["endpoint_name"]
    instance_type = event["instance_type"]    

    print("model_name: \n", model_name)
    print("endpoint_config_name: \n", endpoint_config_name)        
    print("endpoint_name: \n", endpoint_name)                

    ###################################    
    # 엔드 포인트 컨피그 생성
    ###################################
    
    existing_configs = sm_client.list_endpoint_configs(NameContains=endpoint_config_name)['EndpointConfigs']

    if not existing_configs:    
        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1,
                    "InitialInstanceCount": 1,
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )

    existing_endpoints = sm_client.list_endpoints(NameContains=endpoint_name)['Endpoints']

    ###################################
    # 앤드 포인트 생성
    ###################################    
    
    if not existing_endpoints:        
        # 앤드 포인트 생성
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
    else:
        create_endpoint_response = update_sm_endpoint(sm_client, endpoint_name, endpoint_config_name)
        print(f"Endpoint, {endpoint_name}, is updated with endpoint config, {endpoint_config_name}")            

    # 앤드 포인트 상태 정보 추출
    endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']

###########################
# 아래 블록을 넣으면 람다 스템이 타임 아웃으로 실패 함. 현재 이유를 모르겠음.
###########################
    
    # 앤드 포인트가 완료될 때까지 기다림 (약 8분 소요)
    print(f'Endpoint status is creating')    
    while (endpoint_status == 'Creating') | (endpoint_status == 'Updating'):
        endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = endpoint_info['EndpointStatus']
        print(f'Endpoint status: {endpoint_status}')
        if (endpoint_status == 'Creating') | (endpoint_status == 'Updating'):
            time.sleep(20)
            
    if not existing_endpoints:                    
        print(f'Endpoint status is created')                
        return_msg = f"Created Endpoint!"        
    else:
        print(f'Endpoint status is updateed')                
        return_msg = f"Updateed Endpoint!"        
        

    return {
        "statusCode": 200,
        "body": json.dumps(return_msg),
        "other_key": endpoint_name,
    }

def update_sm_endpoint(client, endpoint_name, new_endpoint_config_name):
    '''
    엔드포인트 업데이트 
    '''
    print("endpoint_name: \n", endpoint_name)    
    print("new_endpoint_config_name: ", new_endpoint_config_name)        
    
    current_endpoint_config_name = client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']
    print("current_endpoint_config_name: ", current_endpoint_config_name)    

    
    response = client.update_endpoint(
        EndpointName= endpoint_name,
        EndpointConfigName= new_endpoint_config_name,
    )
    
    print("Swapping is done")
    changed_endpoint_config_name = client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']
    print("changed_endpoint_config_name: \n", changed_endpoint_config_name)        

    
    return response





        
