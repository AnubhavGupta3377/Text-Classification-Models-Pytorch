# Text-Classification-Models
Implementation of State-of-the-art Text Classification Models in Pytorch

## Current Model Implementations
- fastText: Implementation of [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
- TextCNN: Implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

## Usage
1) Download data into "data/" directory or use already available data
2) If using your own data, convert it into the same format as of provided data 
3) Download Pre-trained word embeddings (Glove/Word2Vec) into "data/" directory
4) run following command:

python train.py <path_to_training_file> <path_to_test_file>
