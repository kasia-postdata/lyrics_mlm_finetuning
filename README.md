# Finetuning MLM modesl for songs lyrics (google cloud)

## Dataset

Dataset for finetuning should be in parquet format and contain the column named 'lyrics'. Text will be splited into the lines before finetuning.

## Word2vec

To train or finetune (or both) the Wrod2vec model use this notebook: 'word2vec/lyrics_word2vec.ipynb'


## Transformers

To finetune the transformer model use this script: 'transformers/finetune_mlm.py'


#### Tensorboard

To use tensorboard (google cloud):
Open teminal in Notebook instance and run this command (specify the port numbe and directory with loggs):

`tensorboard --port=8081 --logdir=logdir`

Than open CloudShell and run:

`cloud beta compute ssh --zone "zone-of-your-instance" "name-of-the-notebook-instance" --project "your-project-name-here" -- -L 8081:localhost:8081`

Click Web Preview and choose the right port.
