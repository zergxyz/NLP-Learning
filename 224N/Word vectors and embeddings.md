### Use word vectors to represent the words
---
To perform well on most NLP tasks we first need
to have some notion of similarity and difference between words. With word vectors, we can quite easily encode this ability in the vectors themselves (using distance measures such as Jaccard, Cosine, Euclidean, etc)


### Embeddings:
---
2 type of embeddings: 
* Count based : 1 hot vectors, bag of words, TF-IDF
* vector representations: word2vec, fasttext and others 

one-hot vector: Represent every word as an $\mathbb R ^{V \times 1}$ with all 0s and one 1 at the index of that word in the sorted english language.

![](https://i.imgur.com/iczciFv.png)