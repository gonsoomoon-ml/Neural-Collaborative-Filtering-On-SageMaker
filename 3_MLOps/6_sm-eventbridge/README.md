# Lab6: 6_sm-eventbridge
- EventBridge를 활용하여 CodeBuild config 파일을 레포지토리에 push 하면 code pipeline 을 통해 SageMaker pipeline 실행 (Training).
- EventBridge를 활용하여 Model Approval 하면 code pipeline 을 통해 SageMaker pipeline 실행 (Serving).

# 1. 실습 파일 

- 1.0.Prepare-Dataset.ipynb
    - SageMaker 훈련 Pipeline 실행을 위한 훈련 데이터 준비
- 1.1.Create_eventbridge_for_codepipeline.ipynb
    - EventBridge Rule 및 Target 생성 (+ CodePipeline 실행을 위한 IAM role 및 policy 생성)
- 2.1.Create_eventbridge_for_model_approval.ipynb
    - EventBridge Rule 및 Target 생성 (+ CodePipeline 실행을 위한 IAM role 및 policy 생성)
- 3.1.Cleanup.ipynb
    - 클린업 