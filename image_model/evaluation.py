from transformers import (
    set_seed,
    EarlyStoppingCallback,
)
import os
import shutil
import logging
from utils import Configure
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Normalize
from model.my_models import VitChexpertModel, ResChexpertModel, DenseChexpertModel
from model.metric import accuracy, multi_label_accuracy, macro_f1, micro_f1, auc_threshold, auc
from trainer import Trainer
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", message="adaptive_max_pool2d_backward_cuda does not have")
import time
from data.utils import Data
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

LABEL_NAMES = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion','Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

slurm_job_id = os.environ.get('SLURM_JOB_ID')
print("Slurm job id: {}".format(slurm_job_id))

## reproducibility
seed = 2
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
set_seed(seed)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

def main(resume_checkpoint, parent_dir) -> None:
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

    cfg_path = './config/VitChexpert.cfg'
    args = Configure.get_file_cfg(cfg_path)
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu") ## TODO multi-gpu
    torch.backends.cudnn.benchmark = True

    t = time.time()

    ## TODO check whether thoese are none 
    dataset_val = Data.get_datasets(args.dataset.img_root_path, args.dataset.name, args.dataset.word_idxs, args.dataset.sentsplitter, args.dataset.tokenizer, args.dataset.textfilter,
                                 args.dataset.tokenfilter, args.dataset.max_sent, args.dataset.max_word, multi_image=args.dataset.multi_image,
                                 img_mode=args.dataset.img_trans, img_augment=args.dataset.img_augment, single_test=args.dataset.single_test,
                                 cache_data=args.dataset.cache_dir, section=args.dataset.section, anatomy=None,
                                 meta=None, exclude_ids=None, a_labels=None, split='validation', args = args) ## TODO cleanup
    
    dataset_test = Data.get_datasets(args.dataset.img_root_path, args.dataset.name, args.dataset.word_idxs, args.dataset.sentsplitter, args.dataset.tokenizer, args.dataset.textfilter,
                                 args.dataset.tokenfilter, args.dataset.max_sent, args.dataset.max_word, multi_image=args.dataset.multi_image,
                                 img_mode=args.dataset.img_trans, img_augment=args.dataset.img_augment, single_test=args.dataset.single_test,
                                 cache_data=args.dataset.cache_dir, section=args.dataset.section, anatomy=None,
                                 meta=None, exclude_ids=None, a_labels=None, split='test', args = args) ## TODO cleanup
    
    nw = 0 if args.dataset.cache_dir else args.dataset.num_workers # TODO num_workers=0 if cache_data

    # if args.training.debug:
    #     datasets['train'] = datasets['validation']
    #     # datasets['test'] = torch.utils.data.Subset(datasets['test'], range(2))

    # train_loader = DataLoader(datasets['train'], batch_size=args.dataset.batch_size, shuffle=True, num_workers=nw,
                            #   pin_memory=args.pin_memory)

    batch_size_test = args.dataset.batch_size if args.dataset.batch_size_test is None else args.dataset.batch_size_test
    val_loader = DataLoader(dataset_val, batch_size=batch_size_test, shuffle=False, num_workers=nw,
                            pin_memory=args.pin_memory)
    test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=nw,
                             pin_memory=args.pin_memory)
    

    
    print('Data: validation={0}, test={1} (load time {2:.2f}s)'.format(
                                                                len(val_loader.dataset),
                                                                len(test_loader.dataset),
                                                                time.time() - t))
    
    # logger.info('Data: validation={0}, test={1} (load time {2:.2f}s)'.format(
    #                                                                               len(val_loader.dataset),
    #                                                                               len(test_loader.dataset),
    #                                                                               time.time() - t))
    # train_steps = math.ceil(len(train_loader.dataset) / args.dataset.batch_size) * args.epochs


    # # Parameters
    # params = {'batch_size': 64,
    #         'shuffle': False,
    #         'num_workers': 1} ## TODO num_workers bug

    metric_fns = [multi_label_accuracy, macro_f1, micro_f1]
    criterion = torch.nn.BCEWithLogitsLoss()
    # #TODO Detect last checkpoint
    
    print('Loading checkpoint ...')
    # resume_checkpoint = '/net/scratch/chacha/future_of_work/report-generation/saved/VitChexpertModel/checkpoint-epoch100.pth'
    checkpoint = torch.load(resume_checkpoint)
    state_dict = checkpoint['state_dict']
    # if config['n_gpu'] > 1: ## TODO multiGPU
    #     model = torch.nn.DataParallel(model)
    model = DenseChexpertModel(args.model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # prepare model for testing
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()


    save_csv_name = 'val_new_ddp_metrics.csv'
    bestthrs = cal_metric(None, val_loader, save_csv_name, metric_fns,criterion, model, device, parent_dir)
    save_test_csv_name = 'test_ddp_metrics.csv'
    cal_metric(bestthrs, test_loader, save_test_csv_name, metric_fns,criterion, model, device, parent_dir)



def cal_metric(bestthrs, data_loader, save_csv_name = 'default.csv', metric_fns = [multi_label_accuracy, macro_f1, micro_f1], criterion = torch.nn.BCEWithLogitsLoss(), model = None, device = None, parent_dir = None):
        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))
        labels = []
        outputs = []
        print('Validating ...')
        with torch.no_grad():

            for batch_idx, (ids, img, txt, label, vp) in enumerate(data_loader):
                img, label = img.to(device), label.to(device)
                output, _ = model(img)
                outputs.append(output)
                labels.append(label)

                # save sample images, or do something with output here

                # retrieve report
                # retrieve(output, training_corpus, )
                # get test labels 
                # match from the training set with test labels using similarity function
                #

                # computing loss, metrics on test set
                loss = criterion(output, label)
                batch_size = img.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, label) * batch_size
        
        outputs = torch.cat(outputs, dim=0) 
        labels = torch.cat(labels, dim=0)
        for i, metric in enumerate(metric_fns):
            print(metric.__name__, metric(outputs, labels))
        
        if bestthrs is None:
            bestthrs, f1s = auc_threshold(outputs, labels)
            print('bestthrs', bestthrs)
        
        ## print f1s along with LABEL_NAMES
        # for name, f1 in zip(LABEL_NAMES, f1s):
        #     print(name, f1)
        # print('f1s', f1s)

        acc = multi_label_accuracy(outputs, labels, bestthrs)
        aucs, precision, recall, f1, flatten_acc = auc(outputs, labels, bestthrs)
        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        # logger.info(log)
        epoch = int(resume_checkpoint.split('-epoch')[1].split('.')[0])
        log['epoch'] = epoch 
        log['multi_label_accuracy'] = acc
        log['avg_auc'] = np.mean(aucs)
        log['best_thres_precision'] = np.mean(precision)
        log['best_thres_recall'] = np.mean(recall)
        log['best_thres_f1'] = np.mean(f1)
        for key, value in log.items():
            print(key, value)
        # save metrics to a csv file
        if save_csv_name:
            with open(os.path.join(parent_dir, save_csv_name), 'a') as f:
                f.write(json.dumps(log) + '\n')
        return bestthrs

                
if __name__ == "__main__":
    resume_checkpoint = '/net/scratch/chacha/future_of_work/report-generation/saved/VitChexpertModel/checkpoint-epoch100.pth'
    # parent_dir = '/net/scratch/chacha/future_of_work/report-generation/saved/VitChexpertModel'
    # parent_dir = '/net/scratch/chacha/future_of_work/report-generation/saved/VitChexpertModel/lr_0.5e-3_dense121_ddp'
    parent_dir = '/net/scratch/chacha/future_of_work/report-generation/saved/DenseChexpertModel/lr_1e-5_dense121_ddp_stepLR_20'
    models_list = os.listdir(parent_dir)
    models_list = [el for el in models_list if 'checkpoint' in el]

    ## sorting
    models_list = sorted(models_list, key=lambda x: int(x.split('-epoch')[1].split('.')[0]))
    # models_list = ['model_best.pth']
    for model in models_list:
        print(model)
        resume_checkpoint = os.path.join(parent_dir, model)
        main(resume_checkpoint, parent_dir)
