# Text-Classification-Models
Implementation of State-of-the-art Text Classification Models in Pytorch

## Current Model Implementations
- **fastText:** fastText Model from [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
- **TextCNN:** CNN for text classification proposed in [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- **TextRNN:** Bi-direction LSTM network for text classification

## Upcoming Models
- **RCNN:** Implementation of RCNN Model proposed in [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
- **Seq2seq With Attention:** Implementation of seq2seq model with attention from [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- **Hierarchical Attention:**: Implementation of hierarchical attention model for text classification as proposed in [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

## Usage
1) Download data into "data/" directory or use already available data
2) If using your own data, convert it into the same format as of provided data 
3) Download Pre-trained word embeddings (Glove/Word2Vec) into "data/" directory
4) Go to corresponding model directory
5) run following command:

python train.py <path_to_training_file> <path_to_test_file>

## Model Performance
<table>
  <tr>
    <td>Model</td>
    <td colspan="2">Dataset</td>
  </tr>
  <tr>
    <td></td>
    <td>AG_News</td>
    <td>Query_Well_formedness</td>
  </tr>
  <tr>
    <td>fastText</td>
    <td>89.46</td>
    <td></td>
  </tr>
  <tr>
    <td>TextCNN</td>
    <td>88.57</td>
    <td></td>
  </tr>
  <tr>
    <td>TextRNN</td>
    <td>88.07 (Sequence length = 20) <br/> 90.16 (Flexible sequence length)</td>
    <td></td>
  </tr>
</table>
