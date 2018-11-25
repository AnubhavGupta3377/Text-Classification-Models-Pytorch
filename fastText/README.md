# fastText Model
- Implementation of fastText as proposed in [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)

## Implementation Details
- We used pre-trained Glove Embeddings for encoding words
- Average word embeddings to get sentence embeddings
- Use one hidden layer with 10 hidden units (as described in original paper)
- Feed the output of this layer to a softmax classifier
- Negative Log-Likelihood loss is used
- Used SGD for training

## Model Architecture
![Alt Text](/images/fastText.jpg)
