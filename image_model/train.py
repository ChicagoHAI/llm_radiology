from transformers import set_seed
import os
import logging
from utils import Configure
from model.my_models import VitChexpertModel
from model.metric import multi_label_accuracy, macro_f1, micro_f1
from trainer import Trainer
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", message="adaptive_max_pool2d_backward_cuda does not have")
import time
from data.utils import Data
import torch.utils.data
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

## multi-gpu
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

## WANDB setup
os.environ["WANDB_DIR"] = "/net/scratch/chacha/future_of_work/report-generation/wandb"
os.environ["WANDB_CACHE_DIR"] = "/net/scratch/chacha/future_of_work/report-generation/wandb"
os.environ["WANDB_CONFIG_DIR"] = "/net/scratch/chacha/future_of_work/report-generation/wandb"

## SLURM setup. Feel free to remove this if you are not using SLURM
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

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, args):
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
        # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        # torch.set_deterministic(True)
        # CUDA for PyTorch
    if args.training.single_gpu:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu") ## TODO multi-gpu
        # logger.info('Using device:', str(device))
        torch.backends.cudnn.benchmark = True
    else:
        ddp_setup(rank, world_size)

    # Data
    t = time.time()


    dataset_val = Data.get_datasets(args.dataset.img_root_path, args.dataset.name, args.dataset.word_idxs, args.dataset.sentsplitter, args.dataset.tokenizer, args.dataset.textfilter,
                                 args.dataset.tokenfilter, args.dataset.max_sent, args.dataset.max_word, multi_image=args.dataset.multi_image,
                                 img_mode=args.dataset.img_trans, img_augment=args.dataset.img_augment, single_test=args.dataset.single_test,
                                 cache_data=args.dataset.cache_dir, section=args.dataset.section, anatomy=None,
                                 meta=None, exclude_ids=None, a_labels=None, split='validation', args = args) ## TODO cleanup
    
    nw = 0 if args.dataset.cache_dir else args.dataset.num_workers # TODO num_workers=0 if cache_data

    if args.training.debug:
        dataset_train = dataset_val
    else:
        dataset_train = Data.get_datasets(args.dataset.img_root_path, args.dataset.name, args.dataset.word_idxs, args.dataset.sentsplitter, args.dataset.tokenizer, args.dataset.textfilter,
                            args.dataset.tokenfilter, args.dataset.max_sent, args.dataset.max_word, multi_image=args.dataset.multi_image,
                            img_mode=args.dataset.img_trans, img_augment=args.dataset.img_augment, single_test=args.dataset.single_test,
                            cache_data=args.dataset.cache_dir, section=args.dataset.section, anatomy=None,
                            meta=None, exclude_ids=None, a_labels=None, split='train', args = args) ## TODO cleanup
    

    
    batch_size_test = args.dataset.batch_size if args.dataset.batch_size_test is None else args.dataset.batch_size_test

    if args.training.single_gpu:
        train_loader = DataLoader(dataset_train, batch_size=args.dataset.batch_size, shuffle=True, num_workers=nw,
                                pin_memory=args.pin_memory)
        val_loader = DataLoader(dataset_val, batch_size=batch_size_test, shuffle=False, num_workers=nw,
                                pin_memory=args.pin_memory)
        # test_loader = DataLoader(datasets['test'], batch_size=batch_size_test, shuffle=False, num_workers=nw,
        #                          pin_memory=args.pin_memory)
    else:
        train_loader = DataLoader(dataset_train, batch_size=args.dataset.batch_size, shuffle=False, sampler=DistributedSampler(dataset_train), num_workers=nw,
                                pin_memory=args.pin_memory)
        val_loader = DataLoader(dataset_val, batch_size=batch_size_test, shuffle=False, sampler=DistributedSampler(dataset_val), num_workers=nw,
                                pin_memory=args.pin_memory)
        

    
    print('Data: train={0}, validation={1} (load time {2:.2f}s)'.format(len(train_loader.dataset),
                                                                                  len(val_loader.dataset),
                                                                                #   len(test_loader.dataset),
                                                                                  time.time() - t))


    if args.training.do_validate_data:
        print('Start validating data')
        validate_data(args, val_loader)

        validate_data(args, train_loader)


    if args.training.do_train:
        # Initialize the logger
        logger = logging.getLogger(args.model.arch)
        post_fix = '_debug' if args.training.debug else ''
        saved_dir = os.path.join(args.training.save_dir, args.model.arch + post_fix, args.training.identifier)

        # Initialize the wandb
        if args.training.single_gpu:
            run = wandb.init(project="radiology report generation", name = args.model.arch + post_fix + args.training.identifier)
        else:
            run = wandb.init(project="radiology report generation", name = args.model.arch + post_fix + args.training.identifier, group="DDP")
        # wandb.run.name = args.model.arch + post_fix
        
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir, exist_ok=True)

        logger_dir = os.path.join(saved_dir, 'logs')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir, exist_ok=True)
        
        fh = logging.FileHandler(os.path.join(logger_dir, f'{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'))
        logger.addHandler(fh)
        logger.setLevel(level=logging.DEBUG)
        logger.info('Start logging')
        ## Feel free to remove this if you are not using SLURM
        logger.info("Slurm job id: {}".format(slurm_job_id))
        

        # Training
        print('Start training')
                
        if args.training.single_gpu:
            model = VitChexpertModel(args.model).to(device)
        else:
            model = VitChexpertModel(args.model)
            model = DDP(model.to(rank), device_ids=[rank], output_device=rank)

        # Initialize Trainer
        # get function handles of loss and metrics
        # criterion = getattr(module_loss, config['loss'])
        criterion = torch.nn.BCEWithLogitsLoss()
        metrics = [multi_label_accuracy, macro_f1, micro_f1]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=args.training.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
        # early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.training.patience if args.training.patience else 5)
        if args.training.single_gpu:
            trainer = Trainer(
                        logger,
                        model, 
                        criterion, 
                        metrics, 
                        optimizer,
                        config = args,
                        device = device,
                        data_loader = train_loader,
                        valid_data_loader = val_loader,
                        lr_scheduler = lr_scheduler,
                        wandb = run,
                        save_dir = saved_dir
                        #   callbacks=[early_stopping_callback],
                        )
        else:
            trainer = Trainer(
                            logger,
                            model, 
                            criterion, 
                            metrics, 
                            optimizer,
                            config = args,
                            device = rank,
                            data_loader = train_loader,
                            valid_data_loader = val_loader,
                            lr_scheduler = lr_scheduler,
                            wandb = run,
                            save_dir = saved_dir
                            )

        trainer.train()

        if not args.training.single_gpu:
            destroy_process_group()
        run.finish()


def validate_data(args, val_loader):

    with tqdm(total=len(val_loader.dataset)) as pbar:
        tqdm_interval = 0
        for batch_idx, (ids, img, txt, label, vp) in enumerate(val_loader):
            tqdm_interval += img.shape[0]
            pbar.update(tqdm_interval)

            print(f"Batch {batch_idx + 1}:")
            print(f"  Ids shape: {len(ids)}")
            print(f"  img shape: {img.shape}")
            print(f"  txt shape: {len(txt)}")
            print(f"  label shape: {label.shape}")
            print(f"  vp shape: {vp.shape}")
            
            # Optionally, you can visualize the first image in the batch
            from matplotlib import pyplot as plt
            
            # random sample 10 images from batch size 128
            import random
            random_10_ids = random.sample(range(0, 128), 10)
            images = img[random_10_ids]
            labels = label[random_10_ids]
            for i, image in enumerate(images):
                image = image.numpy().transpose((1, 2, 0))
                image = image * 0.5 + 0.5
                plt.imshow(image)
                plt.savefig('img{0}_{1}_{2}.png'.format(i, ids[random_10_ids[i]], labels[i]))
            break
        
                
if __name__ == "__main__":
    # Get args
    cfg_path = './config/VitChexpert.cfg'
    args = Configure.get_file_cfg(cfg_path)

    if args.training.single_gpu:
        main(0, 1, args)
    else:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, args), nprocs=world_size)

    


