# Code Pipeline 기본 테스트


# 1. 주요 파일 
### 이 워크샵은 ml.p3.2xlarge, ml.p3.8xlarge 세이지 메이커 노트북 인스턴스의 conda_python3 에서 테스트 되었습니다.


- 0.0.Setup_Environment.ipynb
    - 역할 및 권한 필요한 것에 대한 가이드
    - ./CloudFormation/MLOPS-IAM.yaml 을 CloudFormation에서 Stack 배포하여 생성
    - SageMaker 노트북에서 각 Role에 대한 ARN 이 필요함
    
        
- 1.1.create_codecommit.ipynb
    - 코드 리파지토리 생성 및 코드 복사
    
    
- 2.1.build_project.ipynb
    - 빌드 프로젝트 생성


- 3.1.pipeline_project.ipynb
    - 코드 파이프라인 생성


- 4.1.Cleanup.ipynb
    - 클린업 


# 참고: 세이지 메이커 관련
- TBD
