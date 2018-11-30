# fastText Model

This is the implementation of fastText as proposed in [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)

*fastText* is a simple, and yet very powerful model for text classification, proposed by Facebook research. It is one of the fastest text classification models, which given comparable performance to much complex neural network based models.

## Model Architecture
![fastText Architecture](images/fastText.png)

## Implementation Details
- We used pre-trained Glove Embeddings for encoding words
- Average word embeddings to get sentence embeddings
- Use one hidden layer with 10 hidden units (as described in original paper)
- Feed the output of this layer to a softmax classifier
- Negative Log-Likelihood loss is used
- Used SGD for training
