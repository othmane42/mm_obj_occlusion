import pandas as pd
import torch
import os
# import datetime
# import random
# import numpy as np
# import re
# import collections
import sys
from PIL import Image
# Use of __file__ to not be 
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)
from datasets import get_dataset, data_loader , data_cleaning
from torchvision import transforms, utils
from models.get_VCO2_models import LitMMFusionModel
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from pytorch_lightning import callbacks
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
import io
import numpy as np
import torchmetrics
# import logging
# log = logging.getLogger(__name__)

import wandb
os.environ["WANDB_START_METHOD"] = "thread"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HYDRA_FULL_ERROR"]="1"

from omegaconf import DictConfig , OmegaConf
from hydra.utils import instantiate, get_original_cwd, to_absolute_path


# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer #,AutoModel
from utils.seeder import seed_everything


class TestPredStore(Callback):
    def __init__(self,labelencoder,df_x_test,df_y_test,cfg,top_n_rows=100):
        self.test_predictions = []
        self.labelencoder= labelencoder
        self.df_test = df_x_test.copy()
        self.df_test["ground_truth"]=df_y_test
        self.df_test=self.df_test.reset_index(drop=True)
        self.data_table = []
        self.cfg= cfg
        self.top_n_rows= top_n_rows
        self.test_confusion = torchmetrics.classification.ConfusionMatrix(task="multiclass",num_classes=len(self.labelencoder.classes_))

    
    def create_confusion_matrix(self) -> torch.Tensor:
        labels = torch.tensor(self.df_test["ground_truth"])
        preds  = torch.tensor(self.test_predictions)
        
        # confusion matrix
        conf_mat=self.test_confusion(preds,labels).numpy().astype(np.int)
        df_cm = pd.DataFrame(
            conf_mat,
            index=self.labelencoder.classes_,
            columns=self.labelencoder.classes_)
        plt.figure(figsize=[15,15])
        sns.set(font_scale=1)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": "medium"}, fmt='d',linewidths=0.5)
        buf = io.BytesIO()
        
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        return im
        

        

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.test_predictions.extend(outputs.tolist())
    def fill_table(self,row):
        row_value = []
        if self.cfg.dataset.image_column is not None:
            row_value =[wandb.Image(row[self.cfg.dataset.image_column])]
        if self.cfg.dataset.text_columns is not None:
            row_value.extend(row[self.cfg.dataset.text_columns].values.tolist())
        prediction = self.labelencoder.inverse_transform([self.test_predictions[row.name]])[0]
        ground_truth = self.labelencoder.inverse_transform([row["ground_truth"]])[0]
        row_value.extend([prediction,ground_truth])
        
        self.data_table.append(row_value)
        
    def on_test_end(self,trainer,pl_module):
        self.df_test.iloc[:self.top_n_rows,:].apply(lambda x : self.fill_table(x),axis="columns")
        columns= []
        if self.cfg.dataset.image_column is not None:
            columns.append(self.cfg.dataset.image_column)
        if self.cfg.dataset.text_columns is not None:
            columns.extend(self.cfg.dataset.text_columns)
        columns.extend(["prediction","ground_truth"])
        
        im = self.create_confusion_matrix()
        trainer.logger.log_image(key="test_confusion_matrix",images=[im])
        trainer.logger.log_table(key="test eval", columns=columns, data=self.data_table)

 

def is_repeated(model_name,level,dataset_name,cfg):
    # Authenticate with Wandb
    wandb.login()

    # Initialize the API
    api = wandb.Api()

    # Search for the training with team name
    project_name = cfg.wandb.project  # Replace with your search query
    query = {"$and": [{"tags": "fixed_test_size"},
                      {"state":"finished"},
                      {"config.model_name":model_name},
                      {"config.hs_level":level},
                      {"config.dataset":dataset_name},
                     ]}
    #runs = api.runs(f"{team_name}/{project_name}", query)
    runs = api.runs(f"{project_name}", query)

    # Check if any runs were found
    if len(runs) > 0:
        print(f"Found {len(runs)} completed and successful training runs in Wandb .")
        for run in runs:
            #if run.name=="SIMCSE_princeton-nlp/sup-simcse-bert-base-uncased_textcol_Invoice_description_lvl2":
                print(run.name, ":", run.url)
        return True        
    else:
        print(f"No completed and successful training runs found in Wandb .")
        return False



# --- FUNCTIONS ---
def load_models(cfg,num_classes,num_modalities): 
    # --- MODEL ---
    
    encoders= []
     
    # Image encoder 
    if cfg.image_encoder is not None:
        print("Loading image encoder:", cfg.image_encoder.name, end="\n\n")
        image_encoder = instantiate(cfg.image_encoder.init)
        encoders.append(image_encoder)  
     # Text encoder
    if cfg.text_encoder is not None:
        print("Loading text encoder:", cfg.text_encoder.name, end="\n\n")
        text_encoder = instantiate(cfg.text_encoder.init)
        encoders.extend([text_encoder]*len(cfg.dataset.text_columns))
    # Classifier
    print("Loading classifier:", cfg.classifier.name, end="\n\n")
    classifier = instantiate(cfg.classifier.init,num_class=num_classes,hidden_dim = cfg.classifier.init.hidden_dim ,activation_fun = instantiate(cfg.classifier.init.activation_fun),dropout_rate=cfg.classifier.init.dropout_rate)
    fusion=None
    if num_modalities>1:
        # Fusion
        print("Loading fusion:", cfg.fusion.name, end="\n\n")
        keys=list(cfg.fusion.init.keys())
        keys.remove("_target_")
        params = {key : cfg.fusion.init[key] for key in keys }
        if "num_modalities" in keys:
            params["num_modalities"]=num_modalities 
 #        fusion = instantiate(cfg.fusion.init,hidden_dim=cfg.fusion.init.hidden_dim,num_modalities=num_modalities)
        fusion = instantiate(cfg.fusion.init,**params)
        # else:
        #     fusion = instantiate(cfg.fusion.init)
    model = instantiate(cfg.model.init, encoders=encoders, classifier=classifier, fusion=fusion)
  
    return model


   
def train_model(cfg,model,train_dataloader,val_dataloader,test_dataloader,x_test,y_test,labelencoder,classes):    
    model_name=""
    if cfg.text_encoder is not None:
        model_name +=f"{cfg.text_encoder.name}_{cfg.text_encoder.init.checkpoint}_textcol_{';'.join(cfg.dataset.text_columns)}"
        
    if cfg.image_encoder is not None:
        model_name += f"_{cfg.image_encoder.name}"
    if model.fusion is not None:
        fusion_name = cfg.fusion.name
        if cfg.fusion.name=="LowRankTensorFusion":
            fusion_name  = cfg.fusion.name +"_rank_"+ str(cfg.fusion.init.rank)
            print("fusion name",fusion_name)
        model_name +=f"_{fusion_name}"    
    
    res=is_repeated(model_name,cfg.dataset.level,cfg.dataset.name,cfg)
    if res:
        return
    name_experiment = f"{model_name}_lvl{cfg.dataset.level}"
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb_logger = WandbLogger(project=cfg.wandb.project,name = name_experiment,log_model=True , settings=wandb.Settings(start_method="thread"),reinit=True)
    
    model_path = os.path.join(os.getcwd(), f'{cfg.output_folder}{name_experiment}')

    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=model_path, save_top_k=1 ,monitor="val_acc",mode="max")

    early_stopping = callbacks.EarlyStopping(monitor="val_acc",mode="max",patience=10,strict=False)
    
    testPredStore  = TestPredStore(labelencoder,x_test,y_test,cfg)
    
    wandb_logger.experiment.config.update({"model_name": f"{model_name}",
                   "dataset": cfg.dataset.name,
                   "fusion_method":cfg.fusion.name if model.fusion is not None else None,
                    "fusion_params":cfg.fusion.init if model.fusion is not None else None,                       
                   "text_columns": cfg.dataset.text_columns,
                   "image_column":cfg.dataset.image_column,                        
                   "hs_level": cfg.dataset.level,
                   "batch_size": cfg.batch_size,
                   "initial_learning_rate": cfg.model.optimizer.lr,
                   "seed": cfg.seed,
                   "test size":len(test_dataloader)*test_dataloader.batch_size,
                   "train size":len(train_dataloader)*train_dataloader.batch_size,
                   "val size":len(val_dataloader)*val_dataloader.batch_size,
                   "num_classes":len(labelencoder.classes_),
                   "min_value":cfg.dataset.min_value,                         
                   "train_val_size":cfg.dataset.train_val_size,
                   "train_test_size": cfg.dataset.train_test_size,})


    #optim_fun = instantiate(cfg.model.optimizer,params=model.parameters(),lr=cfg.model.optimizer.lr)
    criterion = instantiate(cfg.model.criterion)
    litmodel=LitMMFusionModel(model,classes,criterion,cfg.model.optimizer.lr)
    
    trainer = Trainer(max_epochs=cfg.epochs,fast_dev_run=cfg.fast_dev_run,auto_lr_find=True,deterministic=True,accelerator='gpu', devices=1,logger=wandb_logger,
                      callbacks= [checkpoint_callback,early_stopping,testPredStore])
    
    trainer.fit(litmodel,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
    if not cfg.fast_dev_run:
        print(checkpoint_callback.best_model_path)
        
        # test_model 
        litmodel2=LitMMFusionModel.load_from_checkpoint(checkpoint_callback.best_model_path,model=model)
        results = trainer.test(model=litmodel2, dataloaders=test_dataloader, verbose=True)
    
    
    wandb.finish()
    

    
# --- MAIN --- 
@hydra.main(version_base = "1.3", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # working_dir = os.getcwd()
    # orig_cwd = get_original_cwd()
    # --- SEEDING ---
    seed_everything(cfg.seed)
    if cfg.dataset.text_columns is None and cfg.dataset.image_column is None:
        print("no columns specified ,run aborted")
        return
    # --- OUTPUT_FOLDER ---
    if not os.path.exists(cfg.output_folder):
        os.mkdir(cfg.output_folder)
    # --- DATA AND TRAIN_VAL_TEST SPLIT DATA ---
    file_path = os.path.join(cfg.dataset.root,cfg.dataset.name)
    print("Loading dataset:", cfg.dataset.name, end="\n\n")
    if not os.path.exists(file_path):
        data_cleaning(os.path.join(cfg.dataset.raw_path,cfg.dataset.name),
                                file_path,cfg.dataset.images_folder)
    columns = []
    if cfg.dataset.text_columns is not None:
        columns.extend(cfg.dataset.text_columns)  
    if cfg.dataset.image_column is not None:
        columns.append(cfg.dataset.image_column)
    X_train, y_train, X_val, y_val, X_test, y_test, hs_list, le = get_dataset(file_path,
                                                                         cfg.dataset.min_value,
                                                                         cfg.dataset.level, 
                                                                         cfg.dataset.train_test_size, 
                                                                         cfg.dataset.train_val_size,
                                                                         cfg.seed,drop_na=True,columns=columns )
    
    
    
    # Tokenizers
    
    tokenizers = None
    img_processing=None
    
    index_shifting=0
        
    if cfg.dataset.image_column:
        img_processing ={"transform_fun":instantiate(cfg.image_encoder.image_processing),"params":cfg.image_encoder.params} 
        index_shifting=1
    else:
        cfg.image_encoder = None
    
    if cfg.dataset.text_columns:
        tokenizer = instantiate(cfg.text_encoder.tokenizer)
        tokenizers = []
        for i in range(len(cfg.dataset.text_columns)):
            tokenizers.append((i+index_shifting,{"tokenizer":tokenizer,"params":cfg.text_encoder.params}))   
    else:
        cfg.text_encoder = None
    num_modalities = len(cfg.dataset.text_columns) if cfg.dataset.text_columns is not None else 0
    num_modalities = num_modalities+1 if cfg.dataset.image_column is not None else num_modalities
    
    model=load_models(cfg,len(hs_list),num_modalities)
    
    # --- DATALOADER ---
    #X_train,y_train=X_train.iloc[:32,:],y_train[:32]
    #X_val,y_val=X_val.iloc[:32,:],y_val[:32]
    
    collator = instantiate(cfg.model.collator.init, text_feature_extractors=tokenizers,image_processing=img_processing)
    train_dataloader = data_loader(X_train, y_train, collator, cfg.dataset.text_columns,cfg.batch_size,cfg.dataset.image_column,num_workers=6)
    val_dataloader = data_loader(X_val, y_val, collator,  cfg.dataset.text_columns,cfg.batch_size,cfg.dataset.image_column,shuffle=False,num_workers=6)
    test_dataloader = data_loader(X_test, y_test, collator,  cfg.dataset.text_columns,cfg.batch_size,cfg.dataset.image_column,shuffle=False,num_workers=6)    
    
    
  
    train_model(cfg,model,train_dataloader,val_dataloader,test_dataloader,X_test,y_test,le,hs_list)
    
    
if __name__ == "__main__":
    main()

    