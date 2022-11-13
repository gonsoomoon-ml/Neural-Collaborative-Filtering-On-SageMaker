"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))



def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

# def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
#     try:
#         sm_client = get_sagemaker_client(region)
#         response = sm_client.list_tags(
#             ResourceArn=sagemaker_project_arn)
#         project_tags = response["Tags"]
#         for project_tag in project_tags:
#             new_tags.append(project_tag)
#     except Exception as e:
#         print(f"Error getting project tags: {e}")
#     return new_tags

import os

def print_files_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        print(prefix + path)
        if os.path.isdir(path):
            print_files_in_dir(path, prefix + "    ")

            
def get_pipeline(
    s3_input_data_uri,    
    project_prefix,
    region,
    inference_image_uri = None,
    role=None, # SAGEMAKER_PIPELINE_ROLE_ARN 이 넘어옴.
    default_bucket=None,
    model_package_group_name= None,
    pipeline_name= None,
    base_job_prefix= None,
    # processing_instance_type= None,
    training_instance_type="ml.p3.2xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    ##################################
    ## 입력 파라미터 확인
    ##################################        
    print("######### get_pipeline() input parameter ###############")
    print(f"### BASE_DIR: {BASE_DIR}")    
    print(f"s3_input_data_uri: {s3_input_data_uri}")        
    print(f"project_prefix: {project_prefix}")            
    # print(f"sagemaker_project_arn: {sagemaker_project_arn}")            
    print(f"role: {role}")            
    print(f"default_bucket: {default_bucket}")            
    print(f"model_package_group_name: {model_package_group_name}")            
    print(f"pipeline_name: {pipeline_name}")            
    print(f"base_job_prefix: {base_job_prefix}")                
    # print(f"processing_instance_type: {processing_instance_type}")                
    print(f"training_instance_type: {training_instance_type}")                    
    ##################################
    ## 현재 폴더 기준으로 하위 폴더 및 파일 보기
    ##################################        
    print("######### Look at subfolder and files #########")    
    print_files_in_dir(root_dir =".", prefix="")                    
    
    ##################################
    ## 환경 초기화
    ##################################        
    
    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
    print("role: ", role) # SAGEMAKER_PIPELINE_ROLE_ARN 이 넘어옴.         

    pipeline_session = get_pipeline_session(region, default_bucket)

    ##################################
    ## 소스 코드 다운로드
    ##################################        


    # 데이타의 위치를 code_location.json" 에서 가져온다.
    import json
    json_file_name = "code_location.json"    
    # Opening JSON file
    with open(json_file_name, 'r') as openfile:

        # Reading from json file
        json_object = json.load(openfile)

    print("##### S3 Code Location #########")
    print("s3_code_uri: " , json_object["s3_code_uri"])
    print("################################")    

    
    s3_code_uri = json_object["s3_code_uri"]
    
    
    ##################################
    ## 파이프라인 파라미터 정의
    ##################################        
    
    s3_data_loc = ParameterString(
        name="InputData",
        default_value=s3_input_data_uri,
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    ##################################
    ## 모델 훈련 스텝
    ##################################    

    host_hyperparameters = {'epochs': 1, 
                           'lr': 0.001,
                           'batch_size': 256,
                           'top_k' : 10,
                           'dropout' : 0.0,
                           'factor_num' : 32,
                           'num_layers' : 3,
                           'num_ng' : 4,
                           'test_num_ng' : 99,                   
                        }      
    
    # 훈련 메트릭 정의하여 킆라우드 워치에서 보기
    metric_definitions=[
           {'Name': 'HR', 'Regex': 'HR=(.*?);'},
           {'Name': 'NDCG', 'Regex': 'NDCG=(.*?);'},
           {'Name': 'Loss', 'Regex': 'Loss=(.*?);'}        
        ]

    
    from sagemaker.pytorch import PyTorch

    estimator_output_path = f's3://{default_bucket}/{project_prefix}/training_jobs'
    print("estimator_output_path: \n", estimator_output_path)


    instance_type = 'ml.p3.2xlarge'
    instance_count = 1
        
    # source_dir = s3_code_uri 는 S3 의 경로를 입력한다.
    
    host_estimator = PyTorch(
        entry_point="train.py",           
        source_dir = s3_code_uri,        
        role=role,
        output_path = estimator_output_path,    
        framework_version='1.8.1',
        py_version='py3',
        disable_profiler = True,
        instance_count=instance_count,
        instance_type=instance_type,
        session = pipeline_session, # 세이지 메이커 세션
        hyperparameters=host_hyperparameters,
        metric_definitions = metric_definitions

    )
    
    from sagemaker.inputs import TrainingInput
    from sagemaker.workflow.steps import TrainingStep


    step_train = TrainingStep(
        name= "NCF-Training",
        estimator=host_estimator,
        inputs={
            "train": TrainingInput(
                s3_data= s3_data_loc
            ),
            "test": TrainingInput(
                s3_data= s3_data_loc
            ),        
        }
    )

    ##################################
    ## 모델 아티펙트 리패키징 람다 스텝 정의
    ##################################    
    
    from sagemaker.lambda_helper import Lambda
    from sagemaker.workflow.lambda_step import (
        LambdaStep,
        LambdaOutput,
        LambdaOutputTypeEnum,
    )

    function_name = "sagemaker-lambda-step-repackage-model-artifact"
    repackage_lambda_script_path = f'{BASE_DIR}/iam_repackage_model_artifact.py'
    print("repackage_lambda_script_path: \n", repackage_lambda_script_path)
    
    print("function_name: \n", function_name)
    # Lambda helper class can be used to create the Lambda function
    func_repackage_model = Lambda(
        function_name=function_name,
        execution_role_arn=role,
        script=repackage_lambda_script_path,        
        handler="iam_repackage_model_artifact.lambda_handler",
    )

    output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
    output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
    output_param_3 = LambdaOutput(output_name="S3_Model_URI", output_type=LambdaOutputTypeEnum.String)

    step_repackage_lambda = LambdaStep(
        name="LambdaRepackageStep",
        lambda_func=func_repackage_model,
        inputs={
            "source_path" : s3_code_uri,
            "model_path": step_train.properties.ModelArtifacts.S3ModelArtifacts,
    #        "model_path": artifact_path,        
            "bucket" : default_bucket,
            "prefix" : "ncf/repackage/model"
        },
        outputs=[output_param_1, output_param_2, output_param_3],
    )
    
    
    ##################################
    ## 모델 패키지 그룹 생성 
    ##################################    
    
    import boto3
    sm_client = boto3.client("sagemaker")

    # model_package_group_name = f"{project_prefix}"
    model_package_group_input_dict = {
     "ModelPackageGroupName" : model_package_group_name,
     "ModelPackageGroupDescription" : "Sample model package group"
    }
    response = sm_client.list_model_package_groups(NameContains=model_package_group_name)
    if len(response['ModelPackageGroupSummaryList']) == 0:
        print("No model group exists")
        print("Create model group")    

        create_model_pacakge_group_response = sm_client.create_model_package_group(**model_package_group_input_dict)
        print('ModelPackageGroup Arn : {}'.format(create_model_pacakge_group_response['ModelPackageGroupArn']))    
    else:
        print(f"{model_package_group_name} exitss")    

    ##################################
    ## 모델 등록 스텝
    ##################################            

    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.model import Model

    # inference docker image
#    inference_image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.1-gpu-py3'    

    model = Model(
        image_uri=inference_image_uri,
        model_data = step_repackage_lambda.properties.Outputs["S3_Model_URI"],
        role=role,
        sagemaker_session=pipeline_session,
    )

    register_model_step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.g4dn.xlarge", "ml.p2.xlarge"],
        transform_instances=["ml.g4dn.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
    )

    step_model_registration = ModelStep(
       name="RegisterModel",
       step_args=register_model_step_args,
    )
    
    ##################################
    # pipeline creation
    ##################################    
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            s3_data_loc,
            model_approval_status,
        ],
        steps=[step_train,  step_repackage_lambda, step_model_registration],
        sagemaker_session=pipeline_session,
    )
    return pipeline


