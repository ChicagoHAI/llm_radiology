import numpy as np
import torch
from PIL import Image
import h5py
from torch.utils import data
import pandas as pd
import os
from torchvision import transforms
from utils import Configure, save_preprocessed_data, load_preprocessed_data, normalize
import time
from torchvision.io import read_image
from torchvision.transforms import Normalize
from tqdm import tqdm
from skimage.io import imread, imsave

class CXRDataset(data.Dataset):
    def __init__(self, args, transform=None, split='train'):
        super().__init__()
        self.img_dir_predix = args.dataset.img_dir_predix
        self.args = args
        ## get labels
        label_path = os.path.join(args.dataset.label_path, 'mimic-cxr-2.0.0-chexpert.csv.gz')
        label_data = pd.read_csv(label_path, compression='gzip')
        self.cxr_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion','Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        self.labels = label_data[['subject_id','study_id'] + self.cxr_labels].fillna(0.).replace(-1., 0.)

        split_path = os.path.join(args.dataset.label_path, 'mimic-cxr-2.0.0-split.csv.gz')
        split_data = pd.read_csv(split_path, compression='gzip')

        # study could have multiple dicom jpgs
        self.split_data = split_data[split_data['split']==split][['dicom_id', 'study_id','subject_id']].drop_duplicates() # i need study_id and subject_id
        self.transform = transform

    def __len__(self):
        return self.split_data.shape[0]

    def __getitem__(self, idx):

        t0 = time.time()
        dicom_id, s_id, p_id = self.split_data.iloc[idx]
        label = self.labels[(self.labels['subject_id']==p_id) & (self.labels['study_id']==s_id)][self.cxr_labels].values
        ## change label to torch tensor
        label = torch.tensor(label, dtype=torch.float)
        dicom_id, s_id, p_id = str(dicom_id), str(s_id), str(p_id)
        fpath = os.path.join(self.img_dir_predix, ('p' + p_id[:2]), ('p' + p_id), ('s' + s_id), (dicom_id+'.jpg'))
        
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # fpath = self.files[idx]
        # desired_size = 320
        desired_size = self.args.dataset.input_resolution
        # img = read_image(fpath)
        img = Image.open(fpath)
        t1 = time.time()
        # print("time to open image: ", t1-t0)
        old_size = img.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size, Image.LANCZOS)
        new_img = Image.new('L', (desired_size, desired_size)) # 320x320 L means 8-bit pixels, black and white
        new_img.paste(img, ((desired_size-new_size[0])//2, 
                            (desired_size-new_size[1])//2)) ## center
        img = np.asarray(new_img, np.float64)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0) ## 3 channels TODO i feel like it is unnecessary. its a black and white image why do i need 3 channels
        img = torch.from_numpy(img)
        t2 = time.time()
        # print("time to resize image: ", t2-t1)
        if self.transform:
            img = self.transform(img)
        # t3 = time.time()
        # print("time to transform image: ", t3-t2)
        ##  a zero tensor 
        # txt = torch.zeros(1, 1, 1)
        # txt = torch.random() ## TODO
        
        # label = self.labels[idx]
        img = img.float()
        fpath = os.path.join(self.img_dir_predix, ('p' + p_id[:2]), ('p' + p_id), ('s' + s_id), (dicom_id+'_preprocessed.jpg'))
        print("saving to",fpath)
        imsave(fpath, img)
        
        # try:
        #     assert img.shape == (3, self.args.dataset.input_resolution, self.args.dataset.input_resolution)
        #     # another way to write this is 
        #     # assert img.shape[0]==3 & img.shape[1]== self.args.dataset.input_resolution & img.shape[2]== self.args.dataset.input_resolution
        #     return img, txt, label.squeeze()      
        # except:
        #     print(dicom_id,s_id,p_id)
    
               
if __name__ == "__main__":
    # Get args
    cfg_path = './config/VitChexpert.cfg'
    args = Configure.get_file_cfg(cfg_path)
    
    print("Loading CXRDataset...")
    start = time.time()
    transform = transforms.Compose([transforms.Resize((args.dataset.input_resolution,args.dataset.input_resolution),
                                                        interpolation=transforms.InterpolationMode.BICUBIC), ## downsample image from 320 to input_resolution
                                    Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])
    dset = CXRDataset(args, transform=transform, split='train')
    # start = time.time()
    # dset = CXRPreLoadedDataset(args, transform=transform, split='train')

    
    m_dataloader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False, num_workers=1)
    end = time.time()
    print("Time to load CXRDataset: ", end-start)

    # iterate through the dataloader
    
    ## add a tqdm progress bar
    # import cProfile

    # pr = cProfile.Profile()
    # pr.enable()

    # for i in range(10):
    #     # print(i)
    #     m_dataloader.dataset.__getitem__(i)
    # print("total dataset size", len(m_dataloader.dataset))
    for i, (img, txt, label) in enumerate(tqdm(m_dataloader)):
        # print(img.shape, txt.shape, label.shape)
        print(i, "batch size", img.shape[0])
        if i % 1 == 0:
            print(i)
            end_1k = time.time()
            print(f"Time to load {i} images: ", end_1k-start)
            start = time.time()
            if i == 10:
                break
            # break
    ###Service
    # pr.disable
    # pr.print_stats(sort = 'time')
    # end = time.time()
    # print("Time to load iterate images: ", end-start)


