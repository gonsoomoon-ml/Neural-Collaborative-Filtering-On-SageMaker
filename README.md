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

# 2. 주요 파일 
- ### [중요] 이 워크샵은 ml.p3.2xlarge 세이지 메이커 노트북 인스턴스에서 테스트 되었습니다.


- 0_Setup_Environment
    - 0.0.Setup-Environment.ipynb
        - 필요 패키지 설치 및 로컬 모드 세팅


- 1_Train
    - 1.1.NCF-Train-Scratch.ipynb
        - 세이지 메이커 훈련 잡 없이 훈련 코드를 단계별로 노트북에서 실행
    - 1.2.NCF-Train_Local_Script_Mode.ipynb 
        - 세이지 메이커 로컬 모드,호스트 모드로 훈련 
        - 세이지 메이커 Experiment 사용하여 실험 추적        
    - 1.3.NCF-Train_Horovod.ipynb
        - 세이지 메이커 호로보드 로컬 모드, 호스트 모드로 훈련 


- 2_Inference
    - 2.1.NCF-Inference-Scratch.ipynb
        - 세이지 메이커 배포를 로컬 모드와 호스틀 모드를 단계별 실행
        - 추론을 SageMaker Python SDK 및  Boto3 SDK  구현
    - 2.2.NCF-Inference-SageMaker.ipynb
        - 세이지 메이커 배포 및 서빙을 전체 실행

# 3. 상세 폴더 구성
```
 |-0_Setup_Environment
 | |-0.0.Setup-Environment.ipynb
 | |-daemon.json
 | |-local_mode_setup.sh
 |-1_Train
 | |-1.1.NCF-Train-Scratch.ipynb
 | |-1.3.NCF-Train_Horovod.ipynb
 | |-src
 | | |-train.py
 | | |-train_lib.py
 | | |-train_horovod.py
 | | |-data_utils.py
 | | |-model.py
 | | |-evaluate.py
 | | |-requirements.txt
 | | |-config.py
 |-data
 | |-ml-1m.test.negative
 | |-ml-1m.test.rating
 | |-ml-1m.train.rating
 |-2_Inference
 | |-2.1.NCF-Inference-Scratch.ipynb
 | |-2.2.NCF-Inference-SageMaker.ipynb
 | |-src
 | | |-inference.py
 | | |-inference_utils.py
 | | |-data_utils.py
 | | |-model.py
 | | |-model_config.json
 | | |-common_utils.py
 | | |-requirements.txt
 | | |-config.py


```

# Reference
- [Paper Original Code: Neural Collaborative Filtering)(https://github.com/hexiangnan/neural_collaborative_filtering)
- [원본: 데이터 설명](https://github.com/hexiangnan/neural_collaborative_filtering)

- 세이지 메이커로 파이토치 사용 --> [Use PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)

