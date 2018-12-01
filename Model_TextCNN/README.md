# TextCNN Model
This is the implementation of Convolutional Neural Network for text classification as proposed in the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

CNN has a lot of success applications in the field of Computer Vision. CNNs have recently been successfully applied to NLP problems as well. Text classification is one such problem.

## Model Architecture
![CNNText Architecture](images/TextCNN.png)

## Implementation Details
- Since sentence lengths vary for different sentences, zero pad/truncate each sentence to have length n
- Represent every sentence/text by a 2D tensor of shape (n,d), where d is dimensionality of word vectors
- Used pre-trained Glove Embeddings for encoding words
- Used 3 different filter windows of sizes 3,4,5
- Used 100 feature maps for above windows sizes
- ReLU activaion function
- Max-pooling
- Dropout with keep probability 0.6
- Optimizer - Stochastic Gradient Descent
- Loss function - Negative Log Likelihood
