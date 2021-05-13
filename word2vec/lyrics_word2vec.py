#!/usr/bin/env python
# coding: utf-8

# In[77]:


training_dataset_path = '' # leave empty if using pretrained model, place model in models/ dir
finetuning_dataset_path = 'lyrics_digital_fingerprint.parquet' # dataset suppposed to be in parquet, text should be the last column of dataset
pretrained_model_name = '' # leave empty if you don't have pretrained model
models_dir = 'models/'
n_lines = 1 # 0 to not split lines, 1,2,3 to split lyrics into 1,2,3-lines chunks
num_epochs_ft = 5


# In[61]:


get_ipython().system('pip install gensim==3.8.1')


# In[62]:


get_ipython().system('pip install nltk')


# In[63]:


import pandas as pd
import os
import re
from datetime import datetime
import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
from time import time 
from nltk.tokenize import RegexpTokenizer


# In[64]:


def split_to_n_lines(text, n=1):
    if n > 0:
        lines_iter = iter(text.splitlines())
        lines = []
        for date_time_order in zip(lines_iter):
            lines.append(" ".join(date_time_order))
        return lines
    else: 
        return text


# In[65]:


def load_dataset(dataset_path, n_lines):
    stopwords = {'estribillo', 'x2', 'x3', 'x4', 'x5', 'coro-estribillo', '(x2)', '(x3)', '(x4)', '()',
                 '(repite estribillo 3 veces)'}
    REGEX_PAT = re.compile(r"\b(?:" + ("|").join(stopwords) + ")\\b", re.IGNORECASE)
    df = pd.read_parquet(dataset_path, columns=['lyrics'])
    df["lyrics"] = df["lyrics"].str.replace(REGEX_PAT, "", regex=True)
    df['lyrics'] = df.apply(lambda row: split_to_n_lines(row['lyrics'], n_lines), axis=1)
    df = df.explode('lyrics')
    df['lyrics'] = df.apply(lambda row: (row['lyrics'].strip()), axis=1)
    df = df[df['lyrics'].apply(len) > 10]
    df = df[df['lyrics'].apply(len) < 1000]
    sentences = df['lyrics'].tolist()
    print("Train sentences:", len(sentences))
    return sentences


# In[66]:


def collect_pretrained_model(training_dataset_path, pretrained_model_name, models_dir):
    if pretrained_model_name:
        if os.path.isdir(models_dir + pretrained_model_name):
            model_path = [f for f in listdir(models_dir + pretrained_model_name) if f.endswith('.model')]
            model = gensim.models.Word2Vec.load(model_path[0])
            print('model: ' + model_path[0] + ' loaded')
            return model
        
    elif training_dataset_path:
        training_dataset = load_dataset(training_dataset_path, n_lines)
        training_dataset_tokenized = tokenize_dataset(training_dataset)
        model = gensim.models.Word2Vec(
        training_dataset,
        size=400,
        window=5,
        min_count=2,)
        print('model initialized with data from: ' + training_dataset_path )
        return model
    
    else: 

        if not os.path.isdir(models_dir + 'complete_model/'):
            print("uwaga nie ma foderu chyba bedziemy pobierac")
            os.makedirs(models_dir + 'complete_model/')
            print("Tworze folder")
            url = 'https://zenodo.org/record/1410403/files/complete_model.zip?download=1'
            zip_path, _ = urllib.request.urlretrieve(url)
            with zipfile.ZipFile(zip_path, "r") as f:
                f.extractall(models_dir + 'complete_model/')
        print('aitoralmeida model loaded...')
        return gensim.models.Word2Vec.load(models_dir + 'complete_model/complete.model')
            


# In[67]:


def tokenize_dataset(sentences):
    tokenizer = RegexpTokenizer(r'\w+')
    sentences_tokenized = [w.lower() for w in sentences]
    sentences_tokenized = [tokenizer.tokenize(i) for i in sentences_tokenized]
    return sentences_tokenized


# In[74]:


def finetune_word2vec(model, sentences_tokenized):
    vocab = len(model.wv.vocab)
    print(vocab)
    model.build_vocab(sentences_tokenized, update=True)
    len(model.wv.vocab)
    print(vocab)
    model.train(sentences_tokenized, total_examples=model.corpus_count, epochs=num_epochs_ft)
    print('Your model is finetuned now!!!')
    return model


# In[86]:


def save_model(model, model_path):
    output_dir = "output/{}-{}/".format(model_path.replace("/", "_").replace('.parquet','') , datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(output_dir + "gensim.model")
    model.wv.save_word2vec_format(output_dir + "word2vec.vector")


# In[75]:


model = collect_pretrained_model(training_dataset_path, pretrained_model_name, models_dir)
if finetuning_dataset_path:
    sentences = load_dataset(finetuning_dataset_path, n_lines)
    sentences_tokenized = tokenize_dataset(sentences)
    model_ft = finetune_word2vec(model, sentences_tokenized)


# In[87]:


save_model(model_ft, finetuning_dataset_path)


# In[ ]:





# In[ ]:




