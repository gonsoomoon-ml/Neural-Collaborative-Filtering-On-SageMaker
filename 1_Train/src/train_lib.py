import os
import time
import argparse
import numpy as np
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

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
    #### Setup Environment ##########################    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    
    print("args: \n", args)

    #### Parsing argument  ##########################        
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir    
    model_dir = args.model_dir        
    print("args.train_data_dir: ", train_data_dir)
    print("args.test_data_dir: ", test_data_dir)  
    print("args.model_dir: ", model_dir)  
    
    train_rating_path = os.path.join(train_data_dir, 'ml-1m.train.rating')
    test_negative_path = os.path.join(train_data_dir, 'ml-1m.test.negative')

    
    #### PREPARE DATASET ##########################
    print("=====> data loading <===========")        
    train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_script(train_rating_path, test_negative_path)
    
    # sys.exit()

    # construct the train and test datasets
    print("=====> create data loader <===========")    
    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
            batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)


    
    ########################### CREATE MODEL #################################
    if config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    NCF_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
                            args.dropout, config.model, GMF_model, MLP_model)
    NCF_model.cuda()
    loss_function = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(NCF_model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(NCF_model.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    print("=====> Staring Traiing <===========")
    for epoch in range(args.epochs):
        NCF_model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            NCF_model.zero_grad()
            prediction = NCF_model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        NCF_model.eval()
        HR, NDCG = evaluate.metrics(NCF_model, test_loader, args.top_k)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                    ### Save Model 을 다른 곳에 저장
                    _save_model(NCF_model, model_dir, f'{config.model}.pth')                    
#                 torch.save(NCF_model.state_dict(),'{}{}.pth'.format(config.model_path, config.model))



    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                        best_epoch, best_hr, best_ndcg))


def _save_model(model, model_dir, model_weight_file_name):
    path = os.path.join(model_dir, model_weight_file_name)
    logger.info(f"the model is saved at {path}")    
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)

