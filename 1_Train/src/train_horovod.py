import argparse
import os
import json
import time
import numpy as np
import logging


import sys
# sys.path.append('./src')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist

# Custom Module
import model
import config
import evaluate
from evaluate import ndcg, hit
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



# def metrics(model, test_loader, top_k):
# 	HR, NDCG = [], []

# 	for user, item, label in test_loader:
# 		user = user.cuda()
# 		item = item.cuda()

# 		predictions = model(user, item)
# 		_, indices = torch.topk(predictions, top_k)
# 		recommends = torch.take(
# 				item, indices).cpu().numpy().tolist()

# 		gt_item = item[0].item()
# 		HR.append(hit(gt_item, recommends))
# 		NDCG.append(ndcg(gt_item, recommends))

# 	return np.mean(HR), np.mean(NDCG)


def _metric_average(hvd, val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(hvd, model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    HR_mean = np.mean(HR)
    NDCG_mean = np.mean(NDCG)       
    
    print("len(test_loader.sampler): ", len(test_loader.sampler))

    # Horovod: use test_sampler to determine the number of examples in this worker's partition.
#     HR_mean /= len(test_loader.sampler)
#     NDCG_mean /= len(test_loader.sampler)

    # Horovod: average metric values across workers.
    HR_mean = _metric_average(hvd, HR_mean, "avg_HR")
    NDCG_mean = _metric_average(hvd, NDCG_mean, "avg_NDCG")

    logger.info(
        "Test set: Average HR: {:.4f}, NDCG: {:.2f}%\n".format(HR_mean, NDCG_mean)
    )

    return HR_mean, NDCG_mean

    
def _get_train_data_loader(hvd, args, train_rating_path, test_negative_path, **kwargs):
    logger.info("Get train data sampler and data loader")
    
    print("=====> train data loading <===========")        
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    

    print("=====> create train data loader <===========")    
    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, args.num_ng, True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
    )

    return train_loader, user_num, item_num

def _get_test_data_loader(hvd, args, train_rating_path, test_negative_path, **kwargs):
    logger.info("Get test data sampler and data loader")    
    print("=====> test data loading <===========")        
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    

    print("=====> create test data loader <===========")    
    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_num_ng+1,  sampler=test_sampler, **kwargs
    )

    
    return test_loader




def train_horovod(args):
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
    ## Horovod ##    
    #######################################
    
    import horovod.torch as hvd

    ##  Horovod: initialize library ##
    hvd.init()

    ##  Horovod: pin GPU to local local rank ##
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

    # Horovod: limit number of CPU threads to be used per worker
    torch.set_num_threads(1)

    train_kwargs = {"num_workers": 4, "pin_memory": True}
    test_kwargs = {"num_workers": 1, "pin_memory": True}    

    train_loader, user_num, item_num = _get_train_data_loader(hvd, args, train_rating_path, test_negative_path, **train_kwargs)
    test_loader = _get_test_data_loader(hvd, args, train_rating_path, test_negative_path,  **test_kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    lr_scaler = hvd.size()        
    
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))

    
    #######################################
    ## 데이터 로딩 및 데이터 세트 생성
    #######################################

#     print("=====> data loading <===========")        
#     train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    

#     print("=====> create data loader <===========")    
#     train_dataset = data_utils.NCFData(
#             train_data, item_num, train_mat, args.num_ng, True)
#     test_dataset = data_utils.NCFData(
#             test_data, item_num, train_mat, 0, False)
#     train_loader = data.DataLoader(train_dataset,
#             batch_size=args.batch_size, shuffle=True, num_workers=4)
#     test_loader = data.DataLoader(test_dataset,
#             batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

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

    NCF_model = nn.DataParallel(NCF_model)
    NCF_model.to(device)
    

    #######################################    
    # Horovod: scale learning rate by lr_scaler.
    #######################################    
    loss_function = nn.BCEWithLogitsLoss()    
    
    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(NCF_model.parameters(), lr=args.lr * lr_scaler)
    else:
        optimizer = optim.Adam(NCF_model.parameters(), lr=args.lr * lr_scaler)
    
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(NCF_model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=NCF_model.named_parameters())

    
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

        HR, NDCG = test(hvd, NCF_model, test_loader, args.top_k)
        

        if hvd.rank() == 0:
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

        print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))


def _save_model(model, model_dir, model_weight_file_name):
    path = os.path.join(model_dir, model_weight_file_name)
    print(f"the model is saved at {path}")    
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #### 세이지 메이커 프레임워크의 도커 컨테이너 환경 변수 인자
    ##################################

    parser.add_argument('--train-data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--test-data-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])

    
        
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])    
           
    
    ##################################
    #### 사용자 정의 커맨드 인자
    ##################################

    parser.add_argument("--lr", 
        type=float, 
        default=0.001, 
        help="learning rate")
    parser.add_argument("--dropout", 
        type=float,
        default=0.0,  
        help="dropout rate")
    parser.add_argument("--batch_size", 
        type=int, 
        default=256, 
        help="batch size for training")
    parser.add_argument("--epochs", 
        type=int,
        default=20,  
        help="training epoches")
    parser.add_argument("--top_k", 
        type=int, 
        default=10, 
        help="compute metrics@top_k")
    parser.add_argument("--factor_num", 
        type=int,
        default=32, 
        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", 
        type=int,
        default=3, 
        help="number of layers in MLP model")
    parser.add_argument("--num_ng", 
        type=int,
        default=4, 
        help="sample negative items for training")
    parser.add_argument("--test_num_ng", 
        type=int,
        default=99, 
        help="sample part of negative items for testing")
    parser.add_argument("--out", 
        default=True,
        help="save model or not")
    parser.add_argument("--gpu", 
        type=str,
        default="0",  
        help="gpu card ID")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    
    args = parser.parse_args()
    

    ##################################
    #### 훈련 함수 콜
    ##################################
    
    train_horovod(args)

