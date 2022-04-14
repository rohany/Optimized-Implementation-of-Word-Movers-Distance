# Sinkhorn Word Movers Distance (sinkhorn_wmd)

This repository hosts the source code for an efficient implementation of "Word Mover's Distance" (WMD) using the Sinkhorn-Knopp algorithm. 
Paper reference will be added upon publication.

# To Compile


* source icc compiler 
* source compile

# To Run
first: download the embedding file from https://www.kaggle.com/datasets/yekenot/fasttext-crawl-300d-2m

*Copy the first 100,001 rows in a file called data/vecs.out

*Remove the first line from the file which is 200000 300x

*Then you should be good to go. We do not provide the file, since it is large.

./name_of_executable



## References

- [1] [From Word Embeddings To Document Distances](http://proceedings.mlr.press/v37/kusnerb15.pdf)
- [2] [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/pdf/1306.0895.pdf)
- [3] [Beginner's Guide to Word2Vec and Neural Word Embeddings](https://skymind.ai/wiki/word2vec)
- [4] [Notes on Optimal Transport](https://michielstock.github.io/OptimalTransport/)

