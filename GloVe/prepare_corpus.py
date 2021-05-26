import json
import logging
import os
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
import os
import tarfile
import pandas as pd
import re
import itertools

def progress_bar(t):
#     """ from https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
#     Wraps tqdm instance.
#     Don't forget to close() or __exit__() the tqdm instance once you're done
#     (easiest using `with` syntax).
#     """
#     last_b = [0]

#     def update_to(b=1, bsize=1, tsize=None):
#         """
#         :param b: int, optional
#             Number of blocks transferred so far [default: 1].
#         :param bsize: int, optional
#             Size of each block (in tqdm units) [default: 1].
#         :param tsize: int, optional
#             Total size (in tqdm units). If [default: None] remains unchanged.
#         """
#         if tsize is not None:
#             t.total = tsize
#         t.update((b - last_b[0]) * bsize)
#         last_b[0] = b

#     return update_to


# def download_corpora(urls, datasets_dir):
#     for url in urls:
#         filename = datasets_dir + url.split('/')[-1]
#         if os.path.exists(filename.split('.')[0]):
#             logging.info(f'Corpus {filename}'
#                          f' already downloaded')
#             continue
#         if not os.path.exists(filename):
#             with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
#                       desc=filename) as pb:
#                 urllib.request.urlretrieve(url, filename, reporthook=progress_bar(pb))
#         try:
#             with ZipFile(filename, 'r') as zipObj:
#                 zipObj.extractall(datasets_dir)
#                 os.remove(filename)
#         except:
#             tar = tarfile.open(filename, "r:bz2")
#             for i in tar:
#                 print(i)
#                 tar.extractall(i)
#             tar.close()
#             os.remove(filename)
            
            
def divide_chunks(l, n):
    
    for i in range(0, len(l), n): 
        yield l[i:i + n]
  
  

def read_datasets(datasets_dir):
    datasets_files = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir) if         os.path.isfile(os.path.join(datasets_dir, f))]
    datasets = []
    for dataset in datasets_files:
        print(dataset)

        if dataset.split('.')[-1] == 'json':
            print("Load by json")
            df = pd.read_json(open(dataset))
        else:
            try:
                print("load by csv")
                df = pd.read_csv(open(dataset))
            except:
                print("load by txt")
                with open(dataset) as f:
                    text = f.readlines()
                    df = pd.DataFrame(text, columns=['text'])

        print(df.shape)
        if df.shape[1] == 1:
            df.columns = ['text']
        df = df[df.filter(regex='text').columns]
        df.columns = ['lyrics']
        df = df.dropna()
        df = df[df['lyrics'].apply(len) > 10]
#         df = df[df['lyrics'].apply(len) < 1000]

        try:
            df['lyrics'] = df.apply(lambda row: (row['lyrics'].replace('\n',' ')), axis=1)
        except:
            df['lyrics'] = df.apply(lambda row: (row['lyrics'].str.replace('\n', ' ')), axis=1)
        corpus = df['lyrics'].tolist()
        datasets.append(corpus)
        
    return list(itertools.chain.from_iterable(datasets))



def save_datasets(datasets):
    i = 0
    for corpus in datasets:
        with open(f'datasets/corpus_{i}.out', 'w', encoding='utf8') as f:
            for listitem in corpus:
                f.write(u'{}\n'.format(listitem))
        i = i+1

def main():

    datasets_dir = 'datasets/'
    dataset = read_datasets(datasets_dir)
    datasets = list(divide_chunks(dataset, 10000000))
    print('{} of files created.'.format(len(datasets)))
    save_datasets(datasets)


if __name__ == '__main__':
    main()
