"""Loading dataset useful functions."""

import torch
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder #, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils.seeder import seed_worker
from transformers import models
from PIL import Image
from sklearn.pipeline import Pipeline
from preprocessing.descriptionPreprocessing import TextPreprocessing , CategoryPreprocessing
import collections

COLS = ["Invoice_description","image_name","category","title","categories"]

class CustomDataset(Dataset):
    """CustomDataset with necessary functions for PyTorch.
    Takes input data descriptions (df_X) and target HS codes (df_Y) as well
    as column names used for training (colname1234)."""
    def __init__(self,
                 df_X,
                 df_Y,
                 text_columns = ["Commodity Description of the goods","title","category"],
                 colname_image=None,
                 transform=None,
                ):
        """Init for Custom Dataset."""


        self.df_X = df_X.copy()
        self.df_Y = df_Y.copy()
        self.colname_image=colname_image
        self.text_columns =[] if text_columns is None else text_columns 
        self.transform = transform


    def __getitem__(self,index):
        """Return the input data and target for an index,
        according to the colnames used in __init__."""
        if torch.is_tensor(index):
            index = index.tolist()
        return_elements = []
        if self.colname_image!=None:
            im_path = self.df_X.iloc[index][self.colname_image]
            return_elements.append(im_path)
        label = self.df_Y[index]
        
        for column in self.text_columns:
            if column!=None:
                  return_elements.append(self.df_X.iloc[index][column]) 
        return_elements.append(label)
        # emballé c'est pesé :D
        return (*return_elements,)
    

    def __len__(self):
        """Return length of input dataset."""
        return len(self.df_X)
    

class MMClassificationCollator:
    """Collator Class useful for Pytorch DataLoaders to produce tokenized inputs instead of strings."""
    def __init__(self,text_feature_extractors,image_processing):
        self.text_feature_extractors=text_feature_extractors
        self.image_processing= image_processing
        

    def process_image(self,batch):
        transform_fun, params=self.image_processing["transform_fun"],self.image_processing["params"]
        images=[Image.open(x[0]).convert('RGB') for x in batch]
        if params is not None:
            processed_images = transform_fun(
                images=images,
               **params,
            )
           # processed_images["pixe_values"] = torch.Tensor(processed_images["pixe_values"])
            return processed_images
        elif transform_fun is not None:            
            return transform_fun(images)
        else:
            return images

    def process_texts(self,batch):
        text_encodings=[] 
        for input_idx,config in self.text_feature_extractors:
            tokenizer = config["tokenizer"]
            kwargs = config.get("params", {})
            if tokenizer is not None:
                encoding =tokenizer([x[input_idx] for x in batch], **kwargs)
            else:
                encoding= [x[input_idx] for x in batch]
            text_encodings.append(encoding)
        return (*text_encodings,)
        
    def __call__(self, batch):
        """Collator."""

        output = []
        if self.image_processing is not None:
            image_encoders = self.process_image(batch)
            output.append(image_encoders)
        if self.text_feature_extractors is not None:
            text_encoders=self.process_texts(batch)
            output.extend(text_encoders)
        labels = torch.tensor([x[-1] for x in batch], dtype=torch.long)    
        output.append(labels)
        return (*output,)   
            

def data_cleaning(data_path,output_path,image_folder):


    df_data = pd.read_csv(data_path)

    pipeline_preprocessing=Pipeline(steps=[
         ("desc_processing",TextPreprocessing(mode="classic",column_name="Commodity Description of the goods",
                 special_substitution=True,correct_spelling=True,camel_case_split=True)),
         ("category_preparation",CategoryPreprocessing(column_names=["category","categories"])),
         ('cats_processing',TextPreprocessing(mode="classic",column_name="categories",
                            special_substitution=True,correct_spelling=False,camel_case_split=True,semicolonSep=True)),
         ('cat_processing',TextPreprocessing(mode="classic",column_name="category",
                            special_substitution=True,correct_spelling=False,camel_case_split=True)), 
         ("title_processing",TextPreprocessing(mode="classic",column_name="title",
                 special_substitution=True,correct_spelling=True,camel_case_split=True))
         
         ])   

    df_cleaned=pipeline_preprocessing.transform(df_data)
    df_cleaned=df_cleaned.rename(columns={"HS2":"Proposed_HS2",
                                          "HS4":"Proposed_HS4",
                                          "HS6":"Proposed_HS6",
                                          "Commodity Description of the goods":"Invoice_description"})

    df_cleaned["image_name"] = df_cleaned["image_name"].apply(lambda x : os.path.join(image_folder,x.split("/")[-1]))
    df_cleaned["Invoice_description"].fillna(data['category'],inplace=True)
    df_cleaned.drop_duplicates(subset="Invoice_description",inplace=True) 
    df_cleaned.to_csv(output_path,index=False)



            
def get_dataset(file_path: str, min_value: int, level: int, train_test_size: float, train_val_size: float, seed: int,drop_na=False,columns=None):
        """Load the dataset with parameters and split it into train, val and test.
        Args:
        file_path: path to the dataset file.
        min_value: minimum_value of samples per 'level' to consider as a class.
        level: HS code level (2, 4 or 6 as an integer).
        train_test_size: split size between train and test.
        train_val_size: split size between remaining train and val.
        
        Return: 
        X_train:
        y_train:
        X_val:
        y_val:
        X_test:
        y_test:
        hs_list: list of unique HS{}.format(level) codes considered as classes for training
        based on the minimal number of 'min_value' samples.
        """
        input_df = pd.read_csv(file_path)
        
        # Check the chapter distribution and select the HS{}.format(level) index containing more than 'min_value' data samples for training
        hs_list = input_df["Proposed_HS{}".format(level)].value_counts()[input_df["Proposed_HS{}".format(level)].value_counts()>=min_value].index
        
        # Get the input samples associated with those HS{}
        dataset = input_df[input_df["Proposed_HS{}".format(level)].isin(hs_list)]
        dataset.reset_index(inplace=True)
        if drop_na:
            if columns is not None:
                non_null_data=input_df.dropna(subset=COLS)
                hs_list = non_null_data["Proposed_HS{}".format(level)].value_counts()[non_null_data["Proposed_HS{}".format(level)].value_counts()>=min_value].index
                # Get the input samples associated with those HS{}
                non_null_data = non_null_data[non_null_data["Proposed_HS{}".format(level)].isin(hs_list)]
                y = non_null_data["Proposed_HS{}".format(level)]
                _,X_test,_,y_test=train_test_split(non_null_data,y,test_size=train_test_size,stratify=y,random_state=seed)
                remaining_data=input_df[~input_df.Invoice_description.isin(X_test.Invoice_description)]
                remaining_data=remaining_data.dropna(subset=columns)
                dataset=remaining_data[remaining_data["Proposed_HS{}".format(level)].isin(hs_list)]
                dataset.reset_index(inplace=True)
                le = LabelEncoder()
                X, y = dataset, dataset["Proposed_HS{}".format(level)]
                y = le.fit_transform(dataset["Proposed_HS{}".format(level)])
                y_test = le.transform(y_test)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_val_size, stratify=y, random_state=seed)
                hs_list_test= pd.Series(y_test).value_counts().index
                hs_list_train= pd.Series(y_train).value_counts().index
                hs_list_val= pd.Series(y_val).value_counts().index
                
                assert len(hs_list_train)==len(hs_list_val)==len(hs_list_test), f"make sure the class distribution is equal between train/val/test ! train({len(hs_list_train)}) val({len(hs_list_val)}) test({len(hs_list_test)})"
                return X_train, y_train, X_val, y_val, X_test, y_test, hs_list, le
                        
        X, y = dataset, dataset["Proposed_HS{}".format(level)]
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_size, stratify=y, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=train_val_size, stratify=y_train, random_state=seed)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, hs_list, le
    

def data_loader(X, y, collate_fn, text_columns,
                        batch_size,
                        colname_image=None,
                        transform=None,shuffle=True,num_workers=0):
    """Takes input dataset and colnames and generates
    a PyTorch DataLoader."""
    
    dataset = CustomDataset(X, y, text_columns,
                            colname_image=colname_image,
                            transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=shuffle,
                            worker_init_fn=seed_worker,
                            num_workers=num_workers)
    
    return dataloader
    

  