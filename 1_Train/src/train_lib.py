import os
import time
import numpy as np
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
 
# Custom Module
import model
import config
import evaluate
import data_utils

def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    # https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  

logger = _get_logger()

def train(args):
    '''
    1. args 를 받아서 입력 데이터 로딩
    2. 데이터 세트 생성
    3. 모델 네트워크 생성
    4. 훈련 푸프 실행
    5. 모델 저장
    '''
    #######################################
    ## 환경 확인     
    #######################################
    
    print("args: \n", args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    
    #######################################
    ## 데이타 저장 위치 및 모델 경로 저장 위치 확인
    #######################################
    
    #### Parsing argument  ##########################        
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir    
    model_dir = args.model_dir        
    print("args.train_data_dir: ", train_data_dir)
    print("args.test_data_dir: ", test_data_dir)  
    print("args.model_dir: ", model_dir)  
    
    train_rating_path = os.path.join(train_data_dir, 'ml-1m.train.rating')
    test_negative_path = os.path.join(train_data_dir, 'ml-1m.test.negative')

    
    #######################################
    ## 데이터 로딩 및 데이터 세트 생성 
    #######################################

    print("=====> data loading <===========")        
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    

    print("=====> create data loader <===========")    
    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    
    train_loader = data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=4)
    
#     train_loader = data.DataLoader(train_dataset,
#             batch_size=args.batch_size, shuffle=True, num_workers=4)

    
    test_loader = data.DataLoader(test_dataset,
            batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

    #######################################
    ## 모델 네트워크 생성
    #######################################
    
    if config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
        print("Pretrained model is used")        
    else:
        GMF_model = None
        MLP_model = None
        print("Pretrained model is NOT used")            

    NCF_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
                            args.dropout, config.model, GMF_model, MLP_model)

    NCF_model.to(device)
    
    #######################################
    ## 손실 함수 및 옵티마이저 정의
    #######################################    
    
    loss_function = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(NCF_model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(NCF_model.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    #######################################
    ## 훈련 루프 실행
    #######################################
    
    count, best_hr = 0, 0
    print("=====> Staring Traiing <===========")
    for epoch in range(args.epochs):
        NCF_model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            NCF_model.zero_grad()
            prediction = NCF_model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            count += 1

        NCF_model.eval()
        HR, NDCG = evaluate.metrics(NCF_model, test_loader, args.top_k)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR={:.3f}; \t NDCG={:.3f};".format(np.mean(HR), np.mean(NDCG)))

        print("best_hr: ", best_hr)
        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                ### Save Model 을 다른 곳에 저장
                _save_model(NCF_model, model_dir, f'{config.model}.pth')                    

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                        best_epoch, best_hr, best_ndcg))

    

def _save_model(model, model_dir, model_weight_file_name):
    path = os.path.join(model_dir, model_weight_file_name)
    print(f"the model is saved at {path}")    
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)

