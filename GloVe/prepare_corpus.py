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
    """ from https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    Wraps tqdm instance.
    Don't forget to close() or __exit__() the tqdm instance once you're done
    (easiest using `with` syntax).
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        :param b: int, optional
            Number of blocks transferred so far [default: 1].
        :param bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        :param tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download_corpora(urls, datasets_dir):
    for url in urls:
        filename = datasets_dir + url.split('/')[-1]
        if os.path.exists(filename.split('.')[0]):
            logging.info(f'Corpus {filename}'
                         f' already downloaded')
            continue
        if not os.path.exists(filename):
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=filename) as pb:
                urllib.request.urlretrieve(url, filename, reporthook=progress_bar(pb))
        try:
            with ZipFile(filename, 'r') as zipObj:
                zipObj.extractall(datasets_dir)
                os.remove(filename)
        except:
            tar = tarfile.open(filename, "r:bz2")
            for i in tar:
                print(i)
                tar.extractall(i)
            tar.close()
            os.remove(filename)


def read_datasets(datasets_dir):
    datasets_files = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir) if os.path.isfile(os.path.join(datasets_dir, f))]
    datasets = []
    for dataset in datasets_files:
        try:
            df = pd.read_json(open(dataset))
        except:
            df = pd.read_csv(open(dataset))

        print(dataset)
        print(df.shape)
        df = df[df.filter(regex='text').columns]
        df.columns = ['lyrics']
        df = df.dropna()
        try:
            df['lyrics'] = df.apply(lambda row: (row['lyrics'].replace('\n',' ')), axis=1)
        except:
            df['lyrics'] = df.apply(lambda row: (row['lyrics'].str.replace('\n', ' ')), axis=1)
        datasets.append(df['lyrics'].tolist())
    return list(itertools.chain.from_iterable(datasets))

def save_corpus(corpus):
    with open('corpus', 'w') as f:
        for listitem in corpus:
            f.write('%s\n' % listitem)


def main():
    datasets_url = ['https://zenodo.org/record/3247731/files/preprocessed.tar.bz2?download=1',
                    'https://github.com/linhd-postdata/fandom-lyrics/raw/master/plain_lyrics_dataset.csv.zip']
    datasets_dir = 'datasets/'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    # download_corpora(datasets_url, datasets_dir)
    corpus = read_datasets(datasets_dir)
    print(len(corpus))
    save_corpus(corpus)



if __name__ == '__main__':
    main()
