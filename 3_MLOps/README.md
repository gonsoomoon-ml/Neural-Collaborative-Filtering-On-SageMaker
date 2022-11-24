# MLOps 주요 내용


ML Ops 는 크게 5개의 모듈로 되어 있습니다.크게 아래 3개의 폴더로 구성 되었습니다.
- 전 단계의 1_Train, 2_Inference 의 진행 없이 실습이 가능합니다.

# 0. 사전 작업
## 아래 작업을 이미 하셨다면 스킵 하세요.
- [0_Setup_Environment](../0_Setup_Environment/README.md)
    - 전체 노트북의 시작이며, 필요한 패키지 및 환경 설정을 합니다.
        - 0.0.Setup-Environment.ipynb
    
# 1. 모델 훈련 SageMaker Pipeline
- [1_sm_training_pipeline](1_sm_training_pipeline/README.md)
    - 세이지 메이커 훈련 파이프라인을 구현 함.
        - 1.1.Prepare-Dataset.ipynb
        - 2.1.NCF-Training-Pipeline.ipynb
        
# 2. 모델 서빙 SageMaker Pipeline
- [2_sm_serving_pipeline](2_sm_serving_pipeline/README.md)
    - 세이지 메이커 서빙 파이프라인을 구현 함.
        - 1.1.deployment-pipeline.ipynb
        - 2.1.Inference_Endpoint.ipynb

# 3. Code Pipeline Hello World 실습
- [3_hello-codepipeline/](3_hello-codepipeline/README.md)
    - Code Pipeline 을 노트북에서 실행을 해보는 Hello World 버전 임.
        - 0.0.Setup_Environment.ipynb
        - 1.1.create_codecommit.ipynb
        - 2.1.build_project.ipynb
        - 3.1.pipeline_project.ipynb
        - 4.1.Cleanup.ipynb

# 4. 모델 훈련을 위한 Code Pipeline 을 SageMaker Pipeline 으로 연결
- [4_sm-train-codepipeline/](4_sm-train-codepipeline/README.md)
    - 모델 훈련 파이프라인을 노트북에서 실행을 해보는 실습
        - 1.0.Create_Config.ipynb
        - 1.1.create_codecommit.ipynb
        - 2.1.build_project.ipynb
        - 3.1.pipeline_project.ipynb
        - 4.1.Cleanup.ipynb
        - (옵션) codecommit/sagemaker-pipelines-project.ipynb

# 5. 모델 서빙을 위한 Code Pipeline 을 SageMaker Pipeline 으로 연결
- [5_sm-serving-codepipeline](5_sm-serving-codepipeline/README.md)
    - 모델 추론 파이프라인을 노트북에서 실행을 해보는 실습
        - 1.0.Create_Config.ipynb
        - 1.1.create_codecommit.ipynb
        - 2.1.build_project.ipynb
        - 3.1.pipeline_project.ipynb
        - 4.1.Cleanup.ipynb
        - (옵션) sagemaker-pipelines-project.ipynb

