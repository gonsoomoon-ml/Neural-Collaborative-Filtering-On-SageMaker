# Neural Collaborative Filtering (NCF) On SageMaker

# 1. 배경
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

# 2. 배우게 될 기술 요소들
## 2.1. 세이지 메이커 모델 훈련 및 추론
- 1) 오픈 소스의 코드를 세이지 메이커로 훈련 [1_Train 폴더]
- 2) 세이지 메이커 서빙 [2_Inference 폴더]

## 2.2 MLOps
- 1) ML Ops 로서 SageMaker Pipeline 및 AWS Code Pipeline ]3_MLOps 폴더]

# 3. 실습 방법
## 3.1 [중요 사항]
- #### 이 워크샵은 ml.p3.2xlarge, ml.p3.8xlarge, ml.p3.16xlarge 세이지 메이커 노트북 인스턴스의 `conda_python3`에서 테스트 되었습니다. (세이지 메이커 스튜디오에서 테스트 안됨)

## 3.2. 세이지 메이커 훈련만 하기
- [0_Setup_Environment] 의 0.0.Setup-Environment.ipynb 실행하기
- [1_Train](1_Train/README.md) 폴더를 클릭하시고 README 를 확인 하세요.

## 3.3. 세이지 메이커 훈련 및 추론 하기
아래의 순서대로 세개의 폴더를 진행 합니다.
- [0_Setup_Environment] 의 0.0.Setup-Environment.ipynb 실행하기
- [1_Train](1_Train/README.md) 폴더를 클릭하시고 README 를 확인 하세요.
- [2_Inference](2_Inference/README.md) 폴더를 클릭하시고 README 를 확인 하세요.

## 3.4. MLOps 하기
- [0_Setup_Environment] 의 0.0.Setup-Environment.ipynb 실행하기
- [3_MLOps](3_MLOps/README.md) 폴더를 클릭하시고 README 



### [중요] Clean up
- Cleanup.ipynb
    - 리소스 삭제 입니다. 꼭 확인하시고 지워 주세요. 추가 과금이 발생할 수 있습니다.

# 참고: 논문 관련
- Paper Original Code: Neural Collaborative Filtering
    - 원본 파이토치 코드
    - https://github.com/hexiangnan/neural_collaborative_filtering


- Neural Collaborative Filtering - MLP 실험
    - 한글로 알고리즘에 대해서 잘 설명 됨.
    - https://doheelab.github.io/recommender-system/ncf_mlp/


# 참고: 세이지 메이커 관련

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

