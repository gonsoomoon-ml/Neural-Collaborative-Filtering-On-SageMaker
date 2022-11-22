# Lab3: 3_hello-codepipeline
- Code Pipeline 을 노트북에서 CodeCommit, CodeBuild, 및 CodePipeline 을 Hello World 로 구현 함.

# 1. 실습 파일 

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
