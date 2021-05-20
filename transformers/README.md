# Sentence transformer (XLMRobertaForMaskedLM)


![image](https://user-images.githubusercontent.com/83391529/118826211-913cad00-b8bb-11eb-8971-5c56b1dfc5e8.png)
*It has been shown, that to continue MLM on your own data can improve performances

![image](https://user-images.githubusercontent.com/83391529/119051831-fb407980-b9c3-11eb-81cb-858b26cc1146.png)


papers regarding the model:

https://arxiv.org/abs/2004.09813 (Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation)  
https://arxiv.org/abs/1908.10084 (Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks)  
https://arxiv.org/abs/2004.10964 (Don't Stop Pretraining: Adapt Language Models to Domains and Tasks)  
https://arxiv.org/abs/1911.02116 (Unsupervised Cross-lingual Representation Learning at Scale)  
https://arxiv.org/abs/1907.11692 (RoBERTa: A Robustly Optimized BERT Pretraining Approach)  


## Datasets

We used pretrained model trained on parallel data for 50+ languages and continued pretraining on 
lyrics_digital_fingerprint.parquet
https://github.com/linhd-postdata/fandom-lyrics/blob/master/lyrics_digital_fingerprint.parquet
Size of corpus:
732765 songs
2310276 lines (trained 1 line per sample)
discarding lines shorter than 10 characters

## Training

### Parameters

per_device_train_batch_size = 64  
save_steps = 1000  # Save model every 1k steps  
num_train_epochs = 3  # Number of epochs  
use_fp16 = True  # Set to True, if your GPU supports FP16 operations  
max_length = 20  # Max length for a text input (tokens)  
do_whole_word_mask = True  # If set to true, whole words are masked  
mlm_prob = 15  # Probability that a word is replaced by a [MASK] token  
use_deepspeed = True  

## Evaluation

cos_sim(sentences, base_sentence_embeddings) / original model  
Esta noche voy a cogerte bien 		 He amado, he llorado, he besado, me he entregado 		 Score: 0.3676  
Quiero hincarle el diente a tu culo redondo 		 Esta noche voy a cogerte bien 		 Score: 0.2616  
Quiero hincarle el diente a tu culo redondo 		 He amado, he llorado, he besado, me he entregado 		 Score: 0.2219  
cos_sim(sentences, sentence_embeddings) / our model  
Quiero hincarle el diente a tu culo redondo 		 Esta noche voy a cogerte bien 		 Score: 0.6407  
Esta noche voy a cogerte bien 		 He amado, he llorado, he besado, me he entregado 		 Score: 0.4289  
Quiero hincarle el diente a tu culo redondo 		 He amado, he llorado, he besado, me he entregado 		 Score: 0.4079  
cos_sim(sentences, second_last_sentence_embeddings) / our model (predictions from second and last layer)  
Quiero hincarle el diente a tu culo redondo 		 Esta noche voy a cogerte bien 		 Score: 0.6432  
Quiero hincarle el diente a tu culo redondo 		 He amado, he llorado, he besado, me he entregado 		 Score: 0.5673  
Esta noche voy a cogerte bien 		 He amado, he llorado, he besado, me he entregado 		 Score: 0.5265  

