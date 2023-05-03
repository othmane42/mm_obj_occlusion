# multimodal_framework

this is a framework that contains script for training a multimodal model for hs code prediction. You can launch multi runs using hydra tool as follows : 

```
python3 mm_train.py fast_dev_run=False epochs=50 image_encoder=restnet50 text_encoder=simcse fusion=tensorfusion
```

feel free to edit your command by varying the encoder used for both modalities.
you also have the possibility of running multiple runs as follows: 

```
python mm_train.py image_encoder=vit,clip_image,resnet50 dataset=VCO2_dataset_initial ++dataset.text_columns=[Invoice_description],[Invoice_description,category,title],null ++dataset.level=6 ++classifier.init.hidden_dim=0 fusion=arithmeticfusion,concat,lowranktensorfusion fast_dev_run=False epochs=100 --multirun
```

This instruction will launch a grid search over multiple choise of text columns , image columns , text and image encoders at once , which allows to avoid boilerplate code and increase efficency. 


you also have the possibility to change the dataset by editing your proprer config in the `VCO2/config/dataset` folder. 

## remark

if you encounter any issues with `pyspellchecker` , please follow the following steps : 
```
git clone https://github.com/barrust/pyspellchecker pyspellchecker
cd pyspellchecker
git -c advice.detachedHead=false checkout ae142fac83e9517af5dce476dcfbb24bd20afad8
``` 
