# Language Models for Spanish songs lyrics - experiments

In this repo you will find the scripts to train from scratch or fine-tune models to generate word-level and sentence-level representations:
- Word2vec (https://github.com/aitoralmeida/spanish_word2vec, gensim)
- GloVe (https://github.com/stanfordnlp/GloVe)
- Sentence-BERT (https://github.com/UKPLab/sentence-transformers)

For further instructions please read READ.MD in each of directories:


#### Tensorboard

To use tensorboard (google cloud):
Open teminal in Notebook instance and run this command (specify the port numbe and directory with loggs):

`tensorboard --port=8081 --logdir=logdir`

Than open CloudShell and run:

`cloud beta compute ssh --zone "zone-of-your-instance" "name-of-the-notebook-instance" --project "your-project-name-here" -- -L 8081:localhost:8081`

Click Web Preview and choose the right port.
