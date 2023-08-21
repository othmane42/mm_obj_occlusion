import torch 
import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
from transformers import  AutoTokenizer,AutoModel, ViTModel
from collections import OrderedDict


class VoxNet(nn.Module):
    def __init__(self, n_classes=8, input_shape=(32, 32, 32)):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=6,
                                         out_channels=64, kernel_size=3, padding=1)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 6) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

    def forward(self, x):
        print("x input size is ",x.shape)
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        
        #x = self.mlp(x)
        return x

class VIT(nn.Module):
    def __init__(self,checkpoint: str = 'google/vit-base-patch16-224-in21k',freeze: bool=True):
        super(VIT,self).__init__() 
        self.model = ViTModel.from_pretrained(
            checkpoint,
        )
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self,x):
        # embedding=self.model(**x,return_dict=True).pooler_output
        # return embedding  
         cls_output=self.model(**x,return_dict=True).last_hidden_state[:, 0, :]
         return cls_output

# class ImageEncoder(nn.Module):
#     def __init__(self,model_name) -> None:
#         super(ImageEncoder,self).__init__()
#         self.img2vec = Img2Vec(model=model_name)
    
#     def forward(self,x):
#         pass


class VGG16(nn.Module):
    def __init__(self,freeze:bool=True) -> None:
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = self.model.classifier[:-1]
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False  
 
         
    def forward(self,x):
          image_embeddings = self.model(x)
          return image_embeddings
    

class CustomClassifier(nn.Module):
    def __init__(self,hidden_dim,activation_fun,num_class,dropout_rate=None) -> None:
        super(CustomClassifier,self).__init__()
        self.hidden_dim=hidden_dim
        print("hidden_dim is",hidden_dim)
        if hidden_dim!=0:
            self.fc1 = nn.LazyLinear(hidden_dim)
            self.activation_layer1 = activation_fun
            self.fc2 = nn.Linear(hidden_dim,num_class)
            self.dropout = None
            if dropout_rate:
                self.dropout = nn.Dropout(dropout_rate)
        else:
            self.fc2 = nn.LazyLinear(num_class)

    def forward(self,x):
        if self.hidden_dim!=0:
            x = self.fc1(x)
            x = self.activation_layer1(x)
            if self.dropout:
                x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x



class CLIPImageEncoder(nn.Module):
    CHECKPOINT = "ViT-B/32"
    def __init__(self,checkpoint: str=CHECKPOINT,freeze: bool=True) -> None:
        super(CLIPImageEncoder,self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(checkpoint, device=self.device)

    def forward(self,x):
        if isinstance(x, list):
            images = [self.preprocess(image).unsqueeze(0).to(self.device) for image in x ]
            images=torch.cat(images,0)
        else:
            images = self.preprocess(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            images_features = self.model.encode_image(images)
        return images_features.to(torch.float32)




class ResNet50(nn.Module):
    def __init__(self, output_layer="avgpool",freeze=False):
        super().__init__()
        self.output_layer = output_layer
        pretrained_resnet = models.resnet50(pretrained=True)
        self.children_list = []
        for n,c in pretrained_resnet.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        if freeze:
            for param in self.net.parameters():
                    param.requires_grad = False 
        
    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return x
    


class CustomResNet50(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Modify the first layer to accept custom input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Add a flattening layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        print("in custom resnet x shape is ",x.shape)
        return x

    def get_output_size(self):
        return 256  # Size of the output features