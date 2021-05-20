# Word2vec
Word2Vec is a shallow, two-layer neural networks which is trained to reconstruct linguistic contexts of words.
It does so in one of two ways, either using context to predict a target word (a method known as continuous bag of words, or CBOW), or using a word to predict a target context, which is called skip-gram
![image](https://user-images.githubusercontent.com/83391529/119060091-d18e4f00-b9d1-11eb-83b7-870f6d5920a8.png)

## Training

### Dataset

Our pretrained model can be found here:

### Advantages

- mapps the target word to its context word
- works on the pure co-occurrence probabilities 
- needs little preprocessing, thus requires little memory.

### Disadvantages
- can't separate some opposite word pairs. For example, “good” and “bad” are usually located very close to each other in the vector space, which may limit the performance of word vectors in NLP tasks like sentiment analysis.
- can't deal with out-of-vocabulary words
