import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip
import timm
from torchvision import  models


class ImageEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x
    
class MLPClassier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ImageTextModel(nn.Module):
    def __init__(self, *model_args, **model_kwargs):
        super(ImageTextModel, self).__init__()
        input_size = model_kwargs['input_size']
        hidden_size = model_kwargs['hidden_size']
        num_classes = model_kwargs['num_classes']
        self.image_encoder = ImageEncoder(input_size, hidden_size)
        self.text_encoder = TextEncoder(input_size, hidden_size)
        self.clf = MLPClassier(hidden_size, hidden_size, num_classes)
    
    def forward(self, image, text):
        # encode the image and the text
        encoded_image = self.image_encoder(image)
        encoded_text = self.text_encoder(text)
        return encoded_image, encoded_text

class CLIPModel(nn.Module):
    def __init__(self, input_size, hidden_size, clip_model_name='ViT-B/32'):
        super(CLIPModel, self).__init__(input_size, hidden_size)
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
    
    def forward(self, image, text):
        # encode the image and the text using the parent class
        encoded_image, encoded_text = super().forward(image, text)
        
        # encode the image and text using the CLIP model
        clip_image = self.clip_preprocess(encoded_image).unsqueeze(0).to(device)
        clip_text = self.clip_preprocess(encoded_text).unsqueeze(0).to(device)
        clip_features = self.clip_model.encode_image(clip_image) + self.clip_model.encode_text(clip_text)
        
        return clip_features

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class VitChexpertModel(nn.Module):
    # def __init__(self, input_size, hidden_size, vit_model_name='vit_base_patch16_224'):
    def __init__(self, *model_args, **model_kwargs):
        super(VitChexpertModel, self).__init__()
        # super(VitChexpertModel, self).__init__(*model_args, **model_kwargs)
        ## image_encoder is a vit model
        # vit_model_name= 'resnet50d'
        # self.image_encoder = timm.create_model(vit_model_name, pretrained=True)
        # self.image_encoder = Net()
        # self.image_encoder.reset_classifier(14) ## adhoc
        

        # self.image_encoder = timm.create_model('resnet50d', pretrained=True, num_classes=14, global_pool='catavgmax')
        
        self.image_encoder =  models.densenet121(weights='DEFAULT')
        num_ftrs = self.image_encoder.classifier.in_features
        self.image_encoder.classifier = nn.Sequential(nn.Linear(num_ftrs, 14))
    

        
        # self.image_encoder = timm.create_model('resnetv2_50x1_bitm', pretrained=True, num_classes=14, global_pool='catavgmax')
        # default_cfgs['resnet50']
        # {'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth', 
        # 'num_classes': 1000, 'input_size': (3, 224, 224), 
        # 'pool_size': (7, 7), 'crop_pct': 0.95, 
        # 'interpolation': 'bicubic', 
        # 'mean': (0.485, 0.456, 0.406), 
        # 'std': (0.229, 0.224, 0.225), 
        # 'first_conv': 'conv1', 
        # 'classifier': 'fc'}

        ## image encoder as cnn model


        # self.image_encoder.head = nn.Linear(self.image_encoder.head.in_features, hidden_size)
        ## text_encoder is a dictionary of id to chexpert label
        ## TODO whether to modify the classification head
        self.text_encoder = {}
        ## clf is a MLPClassier
        # self.clf = MLPClassier(hidden_size, hidden_size, num_classes)
    
    def forward(self, image):
        # encode the image and the text using the parent class
        # encoded_image, encoded_text = super().forward(image, text)
        encoded_image = self.image_encoder(image)
        
        ## TODO: encode the text
        text_label = 1
        # self.text_encoder[text]
        # encode the image and text using the CLIP model
        # vit_image = self.vit_model(encoded_image)
        # vit_text = self.vit_model(encoded_text)
        
        return encoded_image, text_label
    


class DenseChexpertModel(nn.Module):
    def __init__(self, *model_args, **model_kwargs):
        super(DenseChexpertModel, self).__init__()        
        self.image_encoder =  models.densenet121(weights='DEFAULT')
        num_ftrs = self.image_encoder.classifier.in_features
        self.image_encoder.classifier = nn.Sequential(nn.Linear(num_ftrs, 14))
        self.text_encoder = {}
   
    def forward(self, image):
        encoded_image = self.image_encoder(image)
        text_label = 1
        return encoded_image, text_label

class ResChexpertModel(nn.Module):
    def __init__(self, *model_args, **model_kwargs):
        super(ResChexpertModel, self).__init__()        
        self.image_encoder = timm.create_model('resnet50d', pretrained=True, num_classes=14, global_pool='catavgmax')
        self.text_encoder = {}
   
    def forward(self, image):
        encoded_image = self.image_encoder(image)
        text_label = 1
        return encoded_image, text_label