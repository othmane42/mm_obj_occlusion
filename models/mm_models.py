import torch 
import pytorch_lightning as pl
from typing import List

import torch.nn as nn
from sklearn.metrics import f1_score, top_k_accuracy_score


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LitMMFusionModel(pl.LightningModule):

    def __init__(self, model,classes,criterion,lr,task="classification"):
        super(LitMMFusionModel,self).__init__()
        self.save_hyperparameters("classes","criterion","lr","task")
        self.model = model
        self.criterion = criterion
        self.lr= lr
        self.classes = classes
        self.task = task

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def __evaluation_metrics(self,output,target,loss,split="train"):
        logs = {}
        y_true = target.detach().cpu().numpy()
        if self.task=="classification":
            y_pred = output.argmax(1).detach().cpu().numpy()
            score = top_k_accuracy_score(y_true, y_pred,k=1)
            top3 = top_k_accuracy_score(y_true, y_pred,k=3)
            top5 = top_k_accuracy_score(y_true, y_pred,k=5)
            logs = {f'{split}_loss': loss, f'{split}_acc': score,f'{split}_top3':top3,f'{split}_top5':top5}
        
        elif self.task=="multilabel":
            y_pred = torch.sigmoid(output).round().detach().cpu().numpy()
            f1_macro= f1_score(y_true, y_pred,average="macro",zero_division=0.0)
            f1_micro = f1_score(y_true, y_pred,average="micro",zero_division=0.0)
            f1_weighted = f1_score(y_true, y_pred,average="weighted",zero_division=0.0)
            f1_samples = f1_score(y_true, y_pred,average="samples",zero_division=0.0)

            logs = {f'{split}_loss': loss, f'{split}_f1_macro': f1_macro,f'{split}_f1_micro':f1_micro,f'{split}_f1_weighted':f1_weighted,f'{split}_f1_samples':f1_samples}    

        return logs

    def training_step(self, batch, batch_idx):
        data_input , target = batch[:-1], batch[-1]
        output = self.model(data_input)
        loss = self.criterion(output, target.float())
        logs=self.__evaluation_metrics(output,target,loss,split="train")
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

 
    def validation_step(self, batch, batch_idx):

        data_input , target = batch[:-1], batch[-1]
        output = self.model(data_input)
        loss = self.criterion(output, target.float())
        logs=self.__evaluation_metrics(output,target,loss,split="val")
        
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):

        data_input , target = batch[:-1], batch[-1]
        output = self.model(data_input)
        loss = self.criterion(output, target.float())
        logs=self.__evaluation_metrics(output,target,loss,split="test")
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return torch.sigmoid(output).round()

   

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': self.optimizer}


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