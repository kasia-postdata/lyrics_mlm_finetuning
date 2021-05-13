#!/usr/bin/env python
# coding: utf-8

# Loading lyrics of the songs


import pandas as pd
import re

stopwords = {'estribillo', 'x2', 'x3', 'x4', 'x5', 'coro-estribillo', '(x2)', '(x3)', '(x4)', '()',
             '(repite estribillo 3 veces)'}
REGEX_PAT = re.compile(r"\b(?:" + ("|").join(stopwords) + ")\\b", re.IGNORECASE)

df = pd.read_parquet("./lyrics_digital_fingerprint.parquet", columns=['spotify_id', 'lyrics'])
df.loc[df["lyrics"].str.contains("estribillo")]
df["lyrics"] = df["lyrics"].str.replace(REGEX_PAT, "", regex=True)

df['lyrics'] = df.apply(lambda row: (row['lyrics'].splitlines()), axis=1)
df = df.explode('lyrics')
df['lyrics'] = df.apply(lambda row: (row['lyrics'].strip()), axis=1)
df = df[df['lyrics'].apply(len) > 10]

from sklearn.model_selection import train_test_split

sentences = df['lyrics'].tolist()
print("Train sentences:", len(sentences))
train_sentences, test_sentences = train_test_split(sentences, test_size=0.15)

print("Train dataset length: " + str(len(train_sentences)))
print("Test dataset length: " + str(len(test_sentences)))


from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from datetime import datetime

model_name = 'sentence-transformers/stsb-xlm-r-multilingual'
per_device_train_batch_size = 64

save_steps = 1000  # Save model every 1k steps
num_train_epochs = 3  # Number of epochs
use_fp16 = True  # Set to True, if your GPU supports FP16 operations
max_length = 20  # Max length for a text input
do_whole_word_mask = True  # If set to true, whole words are masked
mlm_prob = 15  # Probability that a word is replaced by a [MASK] token
use_deepspeed = True

output_dir = "output/{}-{}".format(model_name.replace("/", "_"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Save checkpoints to:", output_dir)

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


# A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True,
                                  max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True,
                                                  max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(test_sentences, tokenizer, max_length, cache_tokenization=True) if len(
    test_sentences) > 0 else None

if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16,
    dataloader_drop_last=True,
    deepspeed='ds_config.json',
    report_to=["tensorboard"],
    warmup_steps=500,
    learning_rate=3e-5,
    logging_dir='tensorflow_logg/'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training done")
