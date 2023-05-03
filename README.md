# multimodal_framework

this is a framework that contains script for training a multimodal model for hs code prediction. You can launch multi runs using hydra tool as follows : 

```
python3 mm_train.py fast_dev_run=False epochs=50 image_encoder=restnet50 text_encoder=simcse fusion=tensorfusion
```

feel free to edit your command by varying the encoder used for both modalities.
you also have the possibility to change the dataset by editing your proprer config in the `VCO2/config/dataset` folder. 
 
