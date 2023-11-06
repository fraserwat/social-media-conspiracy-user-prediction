# Using Sequences of Reddit Posts to Predict User-Level QAnon Participation

This project replicates the findings in a paper predicting a user's participation in conspiracy theory subreddits using past posts. I have expanded on the original paper in three substantial ways:

1. Due to the [growing dominance of PyTorch](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/) I have refactored everything from Tensorflow to PyTorch. As Tensorflow tends to be a more high-level and abstracted library, there were a lot of one-line TF functions or class methods I had to write myself.
1. The LLM-boom of the last year largely happened after the publishing of this paper. I have experimented with other SBERT transformers in an attempt to improve on the original accuracy score.
1. In an effort to improve model interpretability, I have implemented explanatory outputs using Retrieval Augmented Generation (RAG). Our primary model gives a prediction, then our RAG tells us why.

**Original Paper**: http://cs230.stanford.edu/projects_fall_2022/reports/22.pdf

**Paper GitHub Repo:** https://github.com/isvezich/cs230-political-extremism

TODO: Note that when you use padded sequences with LSTM or other RNN layers, you might also want to use packing to make the computation more efficient and accurate, but that's an advanced topic.

### Grid Search Results Word Embedding

```
BEST MLP:
({'l2_penalty_weight': 0.1, 'learning_rate': 0.0001, 'batch_size': 32, 'dropout_rate': 0.1}, {'loss': 0.6109374715731695, 'accuracy': 66.92485755834824, 'recall': 0.578125, 'f1': 0.6348811619033905})

BEST RNN:
({'l2_penalty_weight': 0.1, 'learning_rate': 0.0001, 'batch_size': 32, 'dropout_rate': 0.1}, {'loss': 0.6167022647001804, 'accuracy': 65.00178063527132, 'recall': 0.5764119601328903, 'f1': 0.6107605312330808})

BEST LSTM:
({'l2_penalty_weight': 0.01, 'learning_rate': 0.001, 'batch_size': 16, 'dropout_rate': 0.25}, {'loss': 0.6471295786368383, 'accuracy': 61.85966811396859, 'recall': 0.3864353312302839, 'f1': 0.5268825858642403})
```
