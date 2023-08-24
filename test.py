from models.encoders import VoxNet , CustomResNet50,CustomClassifier
from models.commun_fusion import Concat
from models.mm_models import MMFusion , LitMMFusionModel
import torch 
import torch.nn as nn
from datasets.dataset import data_loader , class_to_index
from pytorch_lightning import Trainer
import os
# useful variable that tells us whether we should use the GPU
use_cuda = torch.cuda.is_available()

if not use_cuda:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...') 
ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FOLDER = os.path.join(ROOT_FOLDER,"small_dataset/")
# Define data folders
image_folder = DATASET_FOLDER+'image'
voxel_folder = DATASET_FOLDER+'velodyne'
label_folder = DATASET_FOLDER+'label'
fast_dev_run= True  
epochs= 10
classes = list(class_to_index.keys())
lr= 0.001
train_data_loader=data_loader(image_folder,voxel_folder,label_folder,batch_size=3) 
image_backbone = CustomResNet50(num_classes=len(class_to_index),
                                input_channels=3)
point_cloud_backbone = VoxNet()
classifier=CustomClassifier(hidden_dim=0,activation_fun=nn.ReLU,
                            num_class=len(class_to_index),dropout_rate=None)
encoders= [image_backbone,point_cloud_backbone]
fusion = Concat()
model_fusion = MMFusion(encoders,fusion,classifier)
#model_path = os.path.join(os.getcwd(), f'{cfg.output_folder}{name_experiment}')

#checkpoint_callback = callbacks.ModelCheckpoint(dirpath=model_path, save_top_k=1 ,monitor="val_acc",mode="max")

#early_stopping = callbacks.EarlyStopping(monitor="val_acc",mode="max",patience=10,strict=False)
optimizer = torch.optim.Adam(model_fusion.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
litmodel=LitMMFusionModel(model_fusion,classes,criterion,lr)

trainer = Trainer(max_epochs=epochs,fast_dev_run=fast_dev_run,
                  auto_lr_find=True,deterministic=True,accelerator='gpu', 
                  devices=1)

trainer.fit(litmodel,train_dataloaders=train_data_loader,val_dataloaders=None)

