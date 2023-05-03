
from sklearn.base import BaseEstimator, TransformerMixin
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
import os
# sys.path.append(os.path.join((os.path.dirname(os.getcwd())),"pyspellchecker/")) 
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)
import re
import unicodedata

from bs4 import BeautifulSoup
import pandas as pd
import swifter
import numpy as np
from pyspellchecker.spellchecker import SpellChecker
from wordsegment import load, segment
import nltk
import ast
load()
ExceptionList = ["3d","2d"]

class TextPreprocessing(BaseEstimator,TransformerMixin):

    def __init__(self,column_name,remove_accents=True,remove_special_chars=True,remove_digits=True,
                    remove_stop_words=True,
                    mode="auto",special_substitution=False, camel_case_split=True,correct_spelling=False,semicolonSep=False):   
        """
        Args:
            mode : text preprossing methods. possible values 
                'classic' for general text preprocessing that includes : removing special chars, accents, digits and lemmatizing tokens ....[TOFILL]
                'auto'  : default model preprocessing is applied.
        """
            
    

        self.column_name = column_name
        self.mode=mode
        self.remove_accents=remove_accents
        self.remove_special_chars=remove_special_chars
        self.remove_digits=remove_digits
        self.remove_stop_words=remove_stop_words
        self.special_substitution = special_substitution
        self.camel_case_split = camel_case_split
        self.correct_spelling = correct_spelling
        self.semicolonSep= semicolonSep


    def _camelCaseSplit(self, text):
        """Split words by camelCase."""
        if text==None:
                return ""
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', str(text))
        return " ".join([m.group(0) for m in matches])

    def _word_split(self,text,spellCheck=False):
        if pd.isna(text) or text=="":
            return ""
        words = segment(text)
        if spellCheck:
            spell = SpellChecker() 
            misspelled=spell.unknown(words)   
            
            #for word in words:
            correction=" ".join([word for word in words 
                    if word not in misspelled  ])
            return correction
        else:
            return self._strip_text(" ".join(words))


    def _strip_text(self,text):
        return re.sub(' +', ' ',text).strip()
        
   
    def _remove_special_characters(self,text,remove_digits=True,semicolon_sep=False):
        pat = "|".join(sorted(map(re.escape, ExceptionList), key=len, reverse=True))
        pattern = re.compile(f'{pat}|([\W0-9]*)', re.S) if not semicolon_sep else re.compile(f'{pat}|([^\w;]*)', re.S) 
        text=pattern.sub(lambda m: " " if m.group(1) else m.group(0), text)
        words=re.findall(r"[\w]+", text) if not semicolon_sep else text.split(";")
        if len(words)==1 and remove_digits:
            text = re.sub("[0-9]+","",text)  
        else:

            for word in words:
                if word in ExceptionList:
                    continue
                matched = re.match("[0-9]+",word)
                is_match = bool(matched)
                if remove_digits and is_match:
                     text=re.sub(r'\b{}\b'.format(word), '', text)
                if len(word)<3:
                    text=re.sub(r'\b{}\b'.format(word), '', text)
                
        return self._strip_text(text)
    

    def _remove_accented_chars(self,text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text 

  

    def _special_substitution(self, text):
        """Replace special cases of text that we do not want to see by sth else."""
        
        pattern = "'s"
        text = re.sub(pattern,' ', text)
        # text = re.sub(pattern,' ', text)
        # text = re.sub(pattern,' ', text)
        pattern = "-" 
        return re.sub(pattern,'', text)
    

  
            



    def _clean_text(self,text):
    
        # if  pd.isna(text):
        #     return None    
        #remove newlines
        if  pd.isna(text):
            return text
        text=re.sub("[\r|\n|\r\n]*",'',text)
        #remove multi spaces
        text=re.sub(' +', ' ',text)
        if self.camel_case_split:
            text = self._camelCaseSplit(text)
        text=text.lower()
        if self.special_substitution:
            text = self._special_substitution(text)
        if self.remove_accents:
            text=self._remove_accented_chars(text)
        if self.remove_special_chars:
            text=re.sub(r"([\[\]{}(-)!\?\.])",r" \1 ",text)
            text=self._remove_special_characters(text,remove_digits=self.remove_digits,semicolon_sep=self.semicolonSep)
        if self.correct_spelling:
            text = self._word_split(text)
        text=self._strip_text(text)
    
            #if Nan, still continue and leave it (we may have blank values for categories/category)
        return text     

    def fit(self,X,y=None):
        return self
     
        
    def transform(self,X,y=None):
        if self.mode=="classic":
            X_copy = X.copy()
            X_copy[self.column_name]=X_copy[self.column_name].swifter.progress_bar(True).apply(lambda x : self._clean_text(x))            
            X_copy["num of words"] = X_copy[self.column_name].apply(lambda x :0  if  type(x)!=str  else len(nltk.word_tokenize(x)))
            to_remove_idx= X_copy[X_copy["num of words"]==0].index
            X_copy[self.column_name].drop(to_remove_idx,inplace=True)

            return X_copy


class CategoryPreprocessing(BaseEstimator,TransformerMixin):
    CATEGS_TO_DELETE = []
    def __init__(self,column_names) -> None:
        super().__init__()
        self.column_names = column_names
    
    def recordsToSemiColonSeparatedString(self,text:str, key:str='name', max_index:int=1963)->pd.DataFrame:
        '''Takes a text composed of records [{key:value},{key:value},...] or  {'value1', 'value2', 'value3', ...} and convert it into semicolon separated string : value;value;value'''
        # Convert string to list of dictionary
        try:
                convertion = ast.literal_eval(text)
                #assert (len(convertion)%2)==0, f"Not a duplicate of categories for {text}!!!!! NOT FITTING !!"
                vals = []
                if isinstance(convertion,list):
                    
                    for j in range(len(convertion)):
                        vals.append(convertion[j][key])
                elif isinstance(convertion,set):
                    for value in convertion:
                        vals.append(value)

                vals=list(dict.fromkeys(vals))
                string= ";".join(vals)     
                
                # Delete the last ";" which is not useful
                return string
                
        except (ValueError,SyntaxError) as e:
                return text
        
    
  
    def camelCaseSplit(self,text):
        """Split words by camelCase."""
        def camelSplit(text):
            matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', str(text))
            return " ".join([m.group(0) for m in matches])
        
        text =  camelSplit(text)
        return text

        
        
    def clearUnusefulCategories(self,text):
        """Delete unuseful categories as 'Home'; 'All Categories'."""
       
        text =text if pd.isna(text) else re.sub(r"(Home;?)|(All Categories;?)|(AllCategories;?)|(^Home)|(^e$)|(N;o;n;e)", "", text) 
        if len(text)>0 and text[-1]==";":
            text = text[:-1] 
        return text
        
        
    def clearEsperluette(self,text):
        """Clear Esperluette for a column in a DataFrame."""
        text = text  if pd.isna(text) else  re.sub(r"&", " ", text) 
        return text


    def clearDoubleSpace(self,text):
        """Clear double space in a string."""
       
        text=re.sub("[\r|\n|\r\n]*",'',text)
        #remove multi spaces
        text=re.sub(' +', ' ',text)
        return text
    
    def clean_function(self,text):
        if  pd.isna(text):
            return text
        text = self.recordsToSemiColonSeparatedString(text)
        text = self.clearUnusefulCategories(text) 
        text = self.clearEsperluette(text)
        text = self.clearDoubleSpace(text)
        text = self.camelCaseSplit(text)
        if text=="" or text==" ":
            return None
        return text  



    def fit(self,X,y=None):
        return self
     
        
    def transform(self,X,y=None):
            X_copy = X.copy()
            for column_name in self.column_names:
                X_copy[column_name]=X_copy[column_name].swifter.progress_bar(True).apply(lambda x : self.clean_function(x))            
            return X_copy




def get_vectors(y_positions,tarbel_dict)-> list:
    y_list = []

    for y in y_positions:
        y_list.append((str(y),tarbel_dict[str(y)]))
    
    return y_list





def load_input(path)->pd.DataFrame:
    df=pd.read_csv(path)
    return df
