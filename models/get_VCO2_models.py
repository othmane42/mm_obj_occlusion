import torch 
import pytorch_lightning as pl
from typing import List

from dataclasses import dataclass
import torch.nn as nn
import torchvision.models as models
from transformers import  AutoTokenizer,AutoModel, ViTModel
from sentence_transformers import SentenceTransformer
import clip

# import our library
import torchmetrics

# initialize metric

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LitMMFusionModel(pl.LightningModule):

    def __init__(self, model,classes,criterion,lr):
        super(LitMMFusionModel,self).__init__()
        self.save_hyperparameters("classes","criterion","lr")
        self.model = model
        self.criterion = criterion
        self.lr= lr
        self.classes = classes
        self.accuracy = torchmetrics.Accuracy(task="multiclass",
                      num_classes=len(self.classes))
        self.top3_accuracy = torchmetrics.Accuracy(task="multiclass",top_k=3,num_classes=len(self.classes))
        self.top5_accuracy = torchmetrics.Accuracy(task="multiclass",top_k=5,num_classes=len(self.classes))
        
        
    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data_input , target = batch[:-1], batch[-1]
        output = self.model(data_input)
        loss = self.criterion(output, target)
        score = self.accuracy(output.argmax(1), target)
        top3 = self.top3_accuracy(output,target)
        top5 = self.top5_accuracy(output,target)
        logs = {'train_loss': loss, 'train_acc': score,"train_top3":top3,"train_top5":top5, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

 
    def validation_step(self, batch, batch_idx):

        data_input , target = batch[:-1], batch[-1]
        output = self.model(data_input)
        loss = self.criterion(output, target)
        score = self.accuracy(output.argmax(1), target)
        top3 = self.top3_accuracy(output,target)
        top5 = self.top5_accuracy(output,target)
        logs = {'val_loss': loss, 'val_acc': score,"val_top3":top3,"val_top5":top5}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):

        data_input , target = batch[:-1], batch[-1]
        output = self.model(data_input)
        loss = self.criterion(output, target)
        score = self.accuracy(output.argmax(1), target)
        top3 = self.top3_accuracy(output,target )
        top5 = self.top5_accuracy(output,target )
        logs = {'test_loss': loss, 'test_acc': score,"test_top3":top3,"test_top5":top5}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return output.argmax(1)

   

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': self.optimizer}


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


class MMFusion(nn.Module):
    def __init__(self,encoders:List,fusion,classifier) -> None:
        super(MMFusion,self).__init__()
        self.encoders = nn.ModuleList(encoders)
        #self.classifier = classifier.to(DEVICE)
        #self.fusion  = fusion.to(DEVICE)
        self.classifier = classifier
        self.fusion  = fusion
 
    def forward(self, x):
     
        encoded=[encoder(x[i][0]) if isinstance(x[i], list) and len(x[i]) == 1 else encoder(x[i]) for i, encoder in enumerate(self.encoders)]
        if self.fusion is not None:
            encoded=self.fusion(encoded)
            out=self.classifier(encoded)
        elif len(encoded)==1:
            out= self.classifier(encoded[0])
        return out


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