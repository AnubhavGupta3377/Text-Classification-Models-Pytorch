# Text-Classification-Models
Implementation of State-of-the-art Text Classification Models in Pytorch

## Current Model Implementations
- **fastText:** fastText Model from [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
- **TextCNN:** CNN for text classification proposed in [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- **TextRNN:** Bi-direction LSTM network for text classification
- **RCNN:** Implementation of RCNN Model proposed in [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
- **CharCNN:** Implementation of character-level CNNs as proposed in the paper [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
- **Seq2seq With Attention:** Implementation of seq2seq model with attention from [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), [Text Classification Research with Attention-based Recurrent Neural Networks](http://univagora.ro/jour/index.php/ijccc/article/download/3142/pdf)
- **Transformer:** Implementation of Transformer model proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Upcoming Models
- **Hierarchical Attention:** Implementation of hierarchical attention model for text classification as proposed in [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

## Requirements
- Python-3.5.0
- Pandas-0.23.4
- Numpy-1.15.2
- Spacy-2.0.13
- Pytorch-0.4.1.post2
- Torchtext-0.3.1

## Usage
1) Download data into "data/" directory or use already available data
2) If using your own data, convert it into the same format as of provided data 
3) Download Pre-trained word embeddings (Glove/Word2Vec) into "data/" directory
4) Go to corresponding model directory
5) run following command:

python train.py <path_to_training_file> <path_to_test_file>

## Model Performance
- All the models were run on a 14GB machine with 2 Cores and one NVIDIA Tesla K80 GPU.
- Runtime in the table below includes the time to load and process the data and running the model.
- Model parameters are not tuned. So, better performance can be achieved by some parameter tuning.

<table>
  <tr>
    <th rowspan="3">Model</th>
    <th align="center" colspan="4">Dataset</th>
  </tr>
  <tr>
    <th colspan="2">AG_News</th>
    <th colspan="2">Query_Well_formedness</th>
  </tr>
  <tr>
    <th>Accuracy </th>
    <th>Runtime </th>
    <th>Accuracy </th>
    <th>Runtime </th>
  </tr>
  <tr>
    <td>fastText</td>
    <td>89.46</td>
    <td>16.0 Mins</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>TextCNN</td>
    <td>88.57</td>
    <td>17.2 Mins</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>TextRNN</td>
    <td>88.07 (Sequence length = 20) <br/> 90.43 (Flexible sequence length)</td>
    <td>21.5 Mins <br/> 36.8 Mins</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>RCNN</td>
    <td>90.61</td>
    <td>22.73 Mins</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>CharCNN</td>
    <td>87.70</td>
    <td>13.08 Mins</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Seq2Seq_Attention</td>
    <td>90.26</td>
    <td>19.10 Mins</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Transformer</td>
    <td>88.49</td>
    <td>47.01 Mins</td>
    <td></td>
    <td></td>
  </tr>
</table>
