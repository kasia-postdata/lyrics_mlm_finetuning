# Word2vec
Word2Vec is a shallow, two-layer neural networks which is trained to reconstruct linguistic contexts of words.
It does so in one of two ways, either using context to predict a target word (a method known as continuous bag of words, or CBOW), or using a word to predict a target context, which is called skip-gram
![image](https://user-images.githubusercontent.com/83391529/119060091-d18e4f00-b9d1-11eb-83b7-870f6d5920a8.png)

## Training
original model: https://github.com/aitoralmeida/spanish_word2vec  
our model:https://zenodo.org/record/1410403  or https://unedo365.sharepoint.com/:f:/s/proyectoercpostdata/EnetU6yNkABBtSezgnDSEnQBQRq_-ByV7UuFlzamJ1_fvg?e=0TJFzN

### Dataset
Original sources:  
- news
-Wikipedia
- Spanish BOE
-web crawling
- open library sources  
total amount of words 3.257.329.900

Our dataset:  
lyrics_digital_fingerprint.parquet (https://github.com/linhd-postdata/fandom-lyrics/blob/master/lyrics_digital_fingerprint.parquet)

Size of corpus:  
732765 songs  
2310276 lines (trained with single lines)  
discarding lines shorter than 10 characters  
trained 1 line per sample  

Our pretrained model can be found here: https://zenodo.org/record/4758900

### Advantages

- mapps the target word to its context word
- works on the pure co-occurrence probabilities 
- needs little preprocessing, thus requires little memory.

### Disadvantages
- can't separate some opposite word pairs. For example, “good” and “bad” are usually located very close to each other in the vector space, which may limit the performance of word vectors in NLP tasks like sentiment analysis.
- can't deal with out-of-vocabulary words
