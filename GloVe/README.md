# GloVe

We used original code from https://github.com/stanfordnlp/GloVe to train from scratch word-level representations of Spanish language.

![image](https://user-images.githubusercontent.com/83391529/118666099-2d4eb180-b7f3-11eb-96c6-3ab00c781403.png)

The GloVe model is trained on the non-zero entries of a global word-word co-occurrence matrix, which tabulates how frequently words co-occur with one another in a given corpus
The training objective of GloVe is to learn word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence.

### Advanteges of model
- semantic similarity between can be measured by cosine similarity or Euclidean distance between word vectors (even for rare words)
- some meaning can be expresed also by vector differences, eg. king − queen = man − woman
- Glove gives lower weight for highly frequent word pairs so as to prevent the meaningless stop words like “the”, “an” will not dominate the training progress.

### Downsides of the model
- only word-level representation
- no representation for out-of-vocabulary words
- to construct co-occurrence matrix of words takes a lot of memory

## Dataset

For the first model we used:
1) Spanish corpora: https://github.com/josecannete/spanish-corpora
Number of lines: 300904000 (300M)
Number of tokens: 2996016962 (3B)
2) Spanish songs lyrics (plain_lyrics_dataset.csv from https://github.com/linhd-postdata/fandom-lyrics/blob/master/plain_lyrics_dataset.csv.zip)
103585 songs
3) Spanish poetry - from Averell (no. 2,3,5,6)
75955 stanzas

For the second smaller model we used only lyrics and poetry ( 2),3) )

## Training

If you want to train model on your own corpora:
1. Place all the files with datasets in `datasets/` directory
  datastes need to be csv or json fromat and name of the column with actual text has to contain 'text' as part of the name or just be a plain text with a document per line 
2. Run:
  `bash train_glove.sh`
  which will generate single corpus fine (from multiple files placed in `datasets/`) with 1 document per line (Cooccurrence contexts for words do not extend past newline characters), tokenize dataset (StantfordTokenizer), and train embeddings.

Tune parameters according to your needs. 

For the first model we used:
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=40
WINDOW_SIZE=4
X_MAX=10

For the first model we used:
VOCAB_MIN_COUNT=5
VECTOR_SIZE=50
MAX_ITER=20
WINDOW_SIZE=4
X_MAX=10

### Evaluation

1. Quiero hincarle el diente a tu culo redondo
2. Esta noche voy a cogerte bien
3. He amado, he llorado, he besado, me he entregado


### requirements

You have to have installed Java 8 and one of the Stanford tools that contains StanfordTokenizer (eg. StanfordParser https://nlp.stanford.edu/software/lex-parser.html )



