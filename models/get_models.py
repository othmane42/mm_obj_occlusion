# Python file loading diverse AI models
import urllib
import zipfile
from typing import List

import fasttext
import numpy as np
import progressbar
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.scripts.glove2word2vec import glove2word2vec
from typing import Union
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from collections import defaultdict
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from simcse import SimCSE
from InferSent.models import InferSent
import torch
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import clip
from typing import List

nltk.download('punkt')

class EmbeddingWrapper(BaseEstimator,TransformerMixin):

    def __init__(self,model_instance: object) -> None:
        assert isinstance(model_instance,SentenceEmbedding) or isinstance(model_instance,WordEmbedding)

        """
        Args:
            model_instance : embedding model instance. possible class are: INFERSENT, EmbeddingWrapper,USE, SentBert , SIMCSE , FasttextEmbedding , GloveEmbedding
        """
        self.model_instance = model_instance

    def transform(self,X,y=None):
       if isinstance(X,list):
            return self.model_instance.encode(X)
       if isinstance(X,pd.Series):
            inputs=X.values.tolist() 
            return self.model_instance.encode(inputs)
       elif isinstance(X,dict):
            inputs=list(X.values())
            vectors = []
            tarbel_dict = defaultdict(list)
            for input_ in inputs:
                input_ = [input_] if isinstance(input_,str) else input_
                output = self.model_instance.encode(input_)
                vectors.append(output)
            
            for tarbel_vector,k in zip(vectors,list(X.keys())):
                 if len(tarbel_vector.shape)<2:
                    tarbel_dict[str(int(k))].append(np.expand_dims(tarbel_vector,axis=0))
                 else:
                    tarbel_dict[str(int(k))] = np.split(tarbel_vector,indices_or_sections=tarbel_vector.shape[0],axis=0)  
            
            return tarbel_dict

    

    def fit(self,X,y=None):
       return self



    def normalize_vecs(self):
        raise NotImplementedError





class WordEmbedding(object):

    def __init__(self,weights_path: str):
       
        if weights_path==None or not os.path.exists(weights_path):
            self.weights_path=self.download_model()
        else:
            self.weights_path = weights_path    
        
    def show_progress(self,block_num, block_size, total_size):
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None

    def download_model(self):
        raise NotImplementedError
    def get_word_vector(self, word: str) -> np.ndarray:
        raise NotImplementedError

    def encode(self, list_sentences: List[str],normalize: bool = False,average: bool =True ) -> np.ndarray:
        vectors = []
        for sentence in list_sentences:
            if sentence=="" or sentence.isspace():
                continue
            words =  word_tokenize(sentence)
            sent_vec = []
            for word in words:
                vector = self.get_word_vector(word)
                if np.isnan(vector).any(): 
                    print(word)
                sent_vec.append(self.div_norm(vector) if normalize else vector )
            sent_vec = np.array(sent_vec)
            sent_vec = self.avg_embedding(sent_vec).squeeze() if  average else sent_vec
           # print(sent_vec.shape,vector.shape)
            vectors.append(sent_vec)
        #print(len(vectors),len(list_sentences),list_sentences[:2],np.vstack(vectors).shape)
        return np.vstack(vectors)

    def l2_norm(self,vector: np.array):
         return np.sqrt(np.sum(vector**2))

    def div_norm(self,sentence_embedding: np.array):
        norm_value = self.l2_norm(sentence_embedding)
        return sentence_embedding * ( 1.0 / norm_value) if norm_value > 0 else sentence_embedding
          

    def avg_embedding(self,embeddings : np.ndarray) -> np.ndarray:
        return np.mean(embeddings,axis=0)




class FasttextEmbedding(WordEmbedding):
    URL ="./cc.en.300.bin"

    def download_model(self):
        fasttext.util.download_model('en', if_exists='ignore')  # English
        

    def __init__(self,weights_path: str = URL) -> None:
        """Word embedding model.
        Args:
            weights_path : path to pretrained model with `.bin` extension
        """

        super().__init__(weights_path)
        self.model = fasttext.load_model(self.weights_path)

        
    # def encode(self, words: List[str]):

    #     return self.model.get_sentence_vector(" ".join(words))

    def get_word_vector(self, word: str) -> np.ndarray:
        return self.model.get_word_vector(word)


class GloveEmbedding(WordEmbedding):
    weight_path = "./glove_word2vec_format.6B.300d.txt"
    URL = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip" # change this to a more useful URL

    def download_model(self):
        response = urllib.request.urlretrieve(self.URL,"./glove.zip",self.show_progress)

        with zipfile.ZipFile("./glove.zip","r") as zip_ref:
                zip_ref.extractall("./glove")

        file_name="./glove_word2vec_format.6B.300d.txt"
        _ = glove2word2vec('./glove/glove.6B.300d.txt',file_name)


        return os.path.abspath(file_name)

    def __init__(self, weights_path: str = URL) -> None:
        """Word embedding model.
        Args:
            weights_path (str): Path to local model. Default: None
        """
        super().__init__(weights_path)

        self.model  = KeyedVectors.load_word2vec_format(self.weights_path)
            
            
    def get_word_vector(self, word: str) -> np.ndarray:
        if word in self.model:
            return self.model[word]
        else:
            return np.random.rand(
                self.model.vector_size,
            )




class SentenceEmbedding(object):

    def encode(self,input: str) -> np.ndarray:
        raise NotImplementedError    


class CLIP(SentenceEmbedding):
    
    def __init__(self):
        self.model, _ = clip.load('ViT-B/32',"cpu")
        self.device = "cpu"

    def encode(self,input_:str) ->np.ndarray:
       #input preparation 
       input_ = clip.tokenize(input_).to(self.device)
       result = self.model.encode_text(input_).squeeze().to(self.device)
       return result.detach().numpy()


class SentBert(SentenceEmbedding):
    weights_path = "bert-base-nli-mean-tokens"  
    def __init__(self,weights_path: str = weights_path):
        self.weights_path = weights_path
        self.model = SentenceTransformer(self.weights_path)
    def encode(self, input: str) -> np.ndarray:
        return self.model.encode(input,show_progress_bar=False)

# class TSDAE(SentenceEmbedding):
#     weights_path = "bert-base-uncased"
    
#     def __init__(self, weights_path:str = weights_path):
#         self.weights_path = weigths_path
#         self.model = 

class USE(SentenceEmbedding):
    weights_path = "https://tfhub.dev/google/universal-sentence-encoder/4" 

    def __init__(self,weights_path:str = weights_path):
        self.weights_path = weights_path
        self.model = hub.load(weights_path)
        
    def encode(self,words)  -> np.ndarray:
        
        return self.model(words).numpy()

class SIMCSE(SentenceEmbedding):
    
    weights_path = "princeton-nlp/sup-simcse-bert-base-uncased"

    def __init__(self,weights_path:str = weights_path):
        self.weights_path = weights_path
        self.model = SimCSE(self.weights_path)

    
    def encode(self,words) -> np.ndarray:
        return self.model.encode(words).numpy()

class INFERSENT(SentenceEmbedding):
        weights_path = 'encoder/infersent2.pkl'
        W2V_PATH = 'fastText/crawl-300d-2M.vec'
        def __init__(self,weights_path: str=weights_path,w2v_path: str=W2V_PATH) -> None:
            self.weights_path = weights_path 
            version = int(weights_path.split(".")[0][-1])
            self.params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
            self.w2v_path = w2v_path
            self.model = InferSent(self.params_model)
            self.model.load_state_dict(torch.load(self.weights_path))   

            #loading vocabularies
            self.model.set_w2v_path(self.w2v_path)
            self.model.build_vocab_k_words(K=100000)


        def encode(self,input) ->np.ndarray:
            return self.model.encode(input, tokenize=True)

class DOC2VEC(SentenceEmbedding):
        weight_path = "d2v.model"
        def __init__(self,weigh_path: str=weight_path) -> None:
            super().__init__()
            self.weight_path = weigh_path
            self.model = Doc2Vec.load(self.weight_path)
        
        def encode(self,input):
            test_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(input)]

            test_vectors = [self.model.infer_vector(doc.words.split(" ")) for doc in test_documents]
            return np.array(test_vectors)

# Perform classification or any other task using the test_vectors



class HSSIMPrediction():
    def __init__(self,model:EmbeddingWrapper,similarity_function,corpus,desc_column,hs_column,count_column) -> None:
        self.model=model
        self.similarity_function=similarity_function
        self.corpus=corpus.copy()
        self.hs_column=hs_column
        self.desc_column= desc_column 
        self.count_column = count_column
        
    def get_top_n(self,similarities,top_n,target_hsCode,normalize):
        
        idx_similarities=np.argsort(similarities,axis=-1)[:,:top_n]
        hs_pred = np.zeros_like(idx_similarities)
        if normalize:
            similarities = (similarities -1 ) * -100
        similarities=np.round(similarities,2)
        desc_pred = []
        hs_pred_data = []
        count_data = []
        sim_data = []
       
        for i,sim in enumerate(idx_similarities):
            hs_pred[i,:] = self.corpus.iloc[sim,:][self.hs_column].values.tolist()
            desc_pred.append(self.corpus.iloc[sim,:][self.desc_column].values.tolist())
            hs_pred_data.append(self.corpus.iloc[sim,:][self.hs_column].values.tolist())
            count_data.append(self.corpus.iloc[sim,:][self.count_column].values.tolist())
            sim_data.append(np.round(similarities[i,sim].tolist(),2) )
        similar_invoices = pd.DataFrame({f"hs_code_top{top_n}":hs_pred_data,f"description_top{top_n}":desc_pred,"num of occurence":count_data,"similarities":sim_data})
        if target_hsCode is not None:
        # you now have k predictions per pixel, and you want that one of them will match the true labels y:
            top_n_score=self.__get_top_k_accuracy(target_hsCode.values,hs_pred)
            return top_n_score,similar_invoices    
        return None , similar_invoices
        

    def get_similar_invoices(self,top_n_range:List,
                            target_description:Union[str,List,pd.Series],
                            target_hsCode,normalize=False):
        target_description = [target_description] if isinstance(target_description,str) else target_description
        print("target embedding ....")
        target_embedding=self.model.transform(target_description)
        print("corpus embedding ....")
        corpus_embeddings=self.model.transform(self.corpus[self.desc_column])
        similarities=self.similarity_function(target_embedding, corpus_embeddings)
        log = {}
        results= None
       
        for k in top_n_range:
            top_n_score,similar_invoices=self.get_top_n(similarities=similarities,top_n=k,target_hsCode=target_hsCode,normalize=normalize)
            if top_n_score is not None:
                log[f"top{k}_accuracy"]= [top_n_score]
            results = similar_invoices if results is None else \
                 pd.concat([results,similar_invoices],axis=1)
            
        if target_hsCode is not None:
            return results, pd.DataFrame(log)
        else:
            return results
    
    
    def __get_top_k_accuracy(self,y_true,y_pred):
            y = torch.Tensor(y_true.copy())
            y_pred =torch.Tensor(y_pred.copy()) 
            correct_pixels = torch.Tensor.float(torch.eq(y[:,None,...], y_pred ).any(dim=1))
            # take the mean of correct_pixels to get the overall average top-k accuracy:
            top_k_acc = correct_pixels.mean()
            return top_k_acc.item()
   



    
    