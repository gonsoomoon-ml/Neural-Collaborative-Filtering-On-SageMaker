import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
from model_def import Net

import sys
import json


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


classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def _metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

    # Horovod: use test_sampler to determine the number of examples in this worker's partition.
    test_loss /= len(test_loader.sampler)
    test_accuracy /= len(test_loader.sampler)

    # Horovod: average metric values across workers.
    test_loss = _metric_average(test_loss, "avg_loss")
    test_accuracy = _metric_average(test_accuracy, "avg_accuracy")

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(test_loss, 100 * test_accuracy)
    )


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def _get_train_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get train data sampler and data loader")
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),        
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )
    return train_loader

def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data sampler and data loader")
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,        
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),        
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=test_batch_size, sampler=test_sampler, **kwargs
    )
    return test_loader


        
## Horovod ##    
import horovod.torch as hvd

def train_horovod(args):
    
    ###############################
    # 커맨드 인자 출력
    ###############################
    logger.debug(f"args: {args}")
    
    ##  Horovod: initialize library ##
    hvd.init()
        
    
    ##  Horovod: pin GPU to local local rank ##
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

    # Horovod: limit number of CPU threads to be used per worker
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True}

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

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

    

    logger.info("Model loaded")
    model = Net()
    
    lr_scaler = hvd.size()    

#     if torch.cuda.device_count() > 1:
#         logger.info("Gpu count: {}".format(torch.cuda.device_count()))
#         model = nn.DataParallel(model)


    # 임시로 사용
    model = nn.DataParallel(model)

    model.cuda()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    ###############################
    # 훈련 시작
    ###############################

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # logger.debug(f"batch_index: {batch_idx}")
            
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            criterion = nn.CrossEntropyLoss()        
            loss = criterion(output, target) # 이와 같이 crossentropy를 이용할 수도 있음.
            
            loss.backward()
            optimizer.step()
            
            
            if batch_idx % args.log_interval == 0:
                
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader)
    
    logger.info("Training is finished")
    return _save_model(model, args.model_dir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )


    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )


    
#     parser.add_argument(
#         "--batch_size", type=int, default=4, metavar="BS", help="batch size (default: 4)"
#     )
#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=0.01,
#         metavar="LR",
#         help="initial learning rate (default: 0.001)",
#     )
#     parser.add_argument(
#         "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
#     )
#     parser.add_argument(
#         "--dist_backend", type=str, default="gloo", help="distributed backend (default: gloo)"
#     )

    #########################
    # Container Environment
    #########################    
    
    #parser.add_argument("--data_dir", type=str, default="Data")    
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    
#    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])    
    
#    parser.add_argument("--model_dir", type=str, default="model")    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])    
    
#    parser.add_argument("--current-host", type=str, default=os.environ["HOST"])    
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    
#    parser.add_argument("--hosts", type=list, default=os.environ["HOST"])    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    
    # parse arguments
    args = parser.parse_args() 
    
    train_horovod(args)
    
