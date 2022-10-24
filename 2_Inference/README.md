# 1. 세이지메이커로 NCF 모델 훈련 및 추론 하기


# 2. 실행 주요 파일 
- #### [중요] 이 워크샵은 ml.p3.2xlarge, ml.p3.8xlarge, ml.p3.16xlarge 세이지 메이커 노트북 인스턴스의 `conda_python3`에서 테스트 되었습니다.


- 0_Setup_Environment
    - 0.0.Setup-Environment.ipynb
        - 필요 패키지 설치 및 로컬 모드 세팅
- 1_Train
    - 1.1.NCF-Train-Scratch.ipynb
        - 세이지 메이커 훈련 잡 없이 훈련 코드를 단계별로 노트북에서 실행
    - 1.2.NCF-Train_Local_Script_Mode.ipynb 
        - 세이지 메이커 로컬 모드,호스트 모드로 훈련 
        - 세이지 메이커 Experiment 사용하여 실험 추적        
    - [옵션] 1.3.NCF-Train_Horovod.ipynb
        - 세이지 메이커 호로보드 로컬 모드, 호스트 모드로 훈련 
    - [옵션] 1.4..NCF-Train_SM_DDP.ipynb
        - 세이지 메이커 Data Parallel Training (DDP) 로 로컬 모드, 호스트 모드로 훈련 
        - [중요] ml.p3.16xlarge 이상의 노트북 인스턴스에서 실행이 가능합니다.
- 2_Inference
    - 2.1.NCF-Inference-Scratch.ipynb
        - 세이지 메이커 배포를 로컬 모드와 호스틀 모드를 단계별 실행
        - 추론을 SageMaker Python SDK 및  Boto3 SDK  구현
    - 2.2.NCF-Inference-SageMaker.ipynb
        - 세이지 메이커 배포 및 서빙을 전체 실행
    * (옵션) 추론 도커 이미지 만들고 추론
        * sagemaker_inference_container/container-inference/
            * 1.1.Build_Docker.ipynb
            * 2.1.Package_Model_Artifact.ipynb
        * 2.3.NCF-Inference-Custom-Docker.ipynb
            * 세이지 메이커 배포 및 서빙을 전체 실행

