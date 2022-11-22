# Neural Collaborative Filtering (NCF) On SageMaker
- 데이타 과학자가 `오픈 소스 코드를 SageMaker 위한 모델 빌딩, 모델 훈련, 모델 서빙` 을 합니다.
- 또한 데이타 과학자에게 어려운 <font color="red">MLOps 를 Jupyter Notebook 에서 구성</font> 할 수 있는 실습을 제공 합니다.

# 0. 배경
NCF 알고리즘을 오픈 소스에 기반하여 SageMaker Training, Serving, 및 MLOps 를 구성을 하는 워크샵 입니다.

# 1. NCF 개요
[Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) 의 논문을
[A pytorch GPU implementation of He et al. "Neural Collaborative Filtering" at WWW'17](https://github.com/guoyang9/NCF) Pytorch 구현 버전이 있습니다.

이 버전은 아래와 같은 성능을 보이고 있으, 원본 논문의 성능과 거의 유사한 결과를 나타냅니다.

 
Models | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10
------ | --------------- | ----------------- | --------------- | -----------------
MLP    | 0.691 | 0.416 | 0.866 | 0.537
GMF    | 0.708 | 0.429 | 0.867 | 0.546
NeuMF (without pre-training) | 0.701 | 0.424 | 0.867 | 0.544
NeuMF (with pre-training)	 | 0.720 | 0.439 | 0.879 | 0.555

이 워크샵은 위의 Pytorch 버전을 SageMaker 에서 훈련 및 서빙을 구현을 했고, SageMaker의 여러가지 장점을 사용하여, 대규모의 데이터 세트에서도 동작할 수 있게 만들었습니다.

# 2. 선행 조건
## 2.1. 이 워크샵을 위한 역할, 권한 설정.
- 아래 경로의 MLOPS-IAM.yaml 파일을 Cloud Formation 에 업로드하여 실행 합니다.
    - [역할 및 권한 설치 YAML 파일](3_MLOps/3_hello-codepipeline/CloudFormation/MLOPS-IAM.yaml)
    
## 2.2. SageMaker Notebook 생성 
### [중요 사항] 
- 이 워크샵은 ml.g4dn.xlarge, ml.p3.2xlarge, ml.p3.8xlarge, ml.p3.16xlarge "세이지 메이커 노트북 인스턴스"의 `conda_python3`에서 테스트 되었습니다.
- SageMaker Notebook 생성시에 Role 선택시에 "<(tack Name)-SageMaker Notebook" 을 꼭 선택해주세요.

## 2.3. 환경 셋업 노트북 실행
- 0_Setup_Environment/0.0.Setup-Environment.ipynb
    - [0_Setup_Environment](0_Setup_Environment/README.md)

# 3. 주요 실습 모듈
## 3.0. [알림] 실습 방법 아래의 3가지 임.
- 3.1.모델 훈련 
    - 소요시간 약 1시간
- 3.1.모델 훈련 과 3.2.모델 서빙 
    - 소요시간 약 1시간 30분
- 3.3. MLOps
    - 소요시간 약 2시간 30분
- 3개의 모듈을 모두할 경우에는 약 4시간이 소요 됨.
    
## 3.1.모델 훈련
- 오픈 소스로 구현된 NCF 알고리즘을 SageMaker 에서 모델 훈련, 호로보드 및 SageMaker DDP (Distributed Data Parallel) 분산 훈련을 함.
    - [모델 훈련: 1_Train](1_Train/README.md)

## 3.2.모델 서빙
- 위에서 모델 훈련된 모델 아티펙트를 모델 서빙을 함. 
    - [모델 서빙: 1_Inference](2_Inference/README.md)

## 3.3. MLOps
- Code Pipeline 읠 쉽게 Jupyter Notebook 에서 실행하면서 배우기
    - [MLOps: 3_MLOps](3_MLOps/README.md)





# 4. [중요] 리소스 정리: Clean up
- Cleanup.ipynb
    - 현재 이 노트북은 작업 중 입니다. 



# A. Reference:
## A.1. 논문 관련
- Paper Original Code: Neural Collaborative Filtering
    - 원본 파이토치 코드
    - https://github.com/hexiangnan/neural_collaborative_filtering
- Neural Collaborative Filtering - MLP 실험
    - 한글로 알고리즘에 대해서 잘 설명 됨.
    - https://doheelab.github.io/recommender-system/ncf_mlp/


## A.2. 세이지 메이커 관련
- 세이지 메이커로 파이토치 사용 
    - [Use PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)
- Use PyTorch with the SageMaker Python SDK
    - https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html
- Amazon SageMaker Local Mode Examples
    - TF, Pytorch, SKLean, SKLearn Processing JOb에 대한 로컬 모드 샘플
        - https://github.com/aws-samples/amazon-sagemaker-local-mode
    - Pytorch 로컬 모드
        - https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/pytorch_script_mode_local_training_and_serving/pytorch_script_mode_local_training_and_serving.py    
- Torch Serve ON AWS
    - https://torchserve-on-aws.workshop.aws/en/150.html
- Pytorch weight 저장에 대해 우리가 알아야하는 모든 것
    - https://comlini8-8.tistory.com/50
- Pytorch Dataset과 Dataloader 및 Sampler    
    - https://gaussian37.github.io/dl-pytorch-dataset-and-dataloader/    

