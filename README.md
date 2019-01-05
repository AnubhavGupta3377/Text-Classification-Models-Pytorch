# Text-Classification-Models-Pytorch
Implementation of State-of-the-art Text Classification Models in Pytorch

## Implemented Models
- **fastText:** fastText Model from [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
- **TextCNN:** CNN for text classification proposed in [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- **TextRNN:** Bi-direction LSTM network for text classification
- **RCNN:** Implementation of RCNN Model proposed in [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
- **CharCNN:** Implementation of character-level CNNs as proposed in the paper [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
- **Seq2seq With Attention:** Implementation of seq2seq model with attention from [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), [Text Classification Research with Attention-based Recurrent Neural Networks](http://univagora.ro/jour/index.php/ijccc/article/download/3142/pdf)
- **Transformer:** Implementation of Transformer model proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

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
    <th>Accuracy (%)</th>
    <th>Runtime </th>
    <th>Accuracy (%)</th>
    <th>Runtime </th>
  </tr>
  <tr>
    <td>fastText</td>
    <td>89.46</td>
    <td>16.0 Mins</td>
    <td>62.10</td>
    <td>7.0 Mins</td>
  </tr>
  <tr>
    <td>TextCNN</td>
    <td>88.57</td>
    <td>17.2 Mins</td>
    <td>67.38</td>
    <td>7.43 Mins</td>
  </tr>
  <tr>
    <td>TextRNN</td>
    <td>88.07 (Seq len = 20) <br/> 90.43 (Flexible seq len)</td>
    <td>21.5 Mins <br/> 36.8 Mins</td>
    <td>68.29 <br/> 66.29</td>
    <td>7.69 Mins <br/> 7.25 Mins</td>
  </tr>
  <tr>
    <td>RCNN</td>
    <td>90.61</td>
    <td>22.73 Mins</td>
    <td>66.70</td>
    <td>7.21 Mins</td>
  </tr>
  <tr>
    <td>CharCNN</td>
    <td>87.70</td>
    <td>13.08 Mins</td>
    <td>68.83</td>
    <td>2.49 Mins</td>
  </tr>
  <tr>
    <td>Seq2Seq_Attention</td>
    <td>90.26</td>
    <td>19.10 Mins</td>
    <td>67.84</td>
    <td>7.36 Mins</td>
  </tr>
  <tr>
    <td>Transformer</td>
    <td>88.54</td>
    <td>46.47 Mins</td>
    <td>63.43</td>
    <td>5.77 Mins</td>
  </tr>
</table>

## References
[1] [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759) </br>
[2] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) </br>
[3] [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552) </br>
[4] [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) </br>
[5] [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) </br>
[6] [Text Classification Research with Attention-based Recurrent Neural Networks](http://univagora.ro/jour/index.php/ijccc/article/download/3142/pdf) </br>
[7] [Attention Is All You Need](https://arxiv.org/abs/1706.03762) </br>
[8] [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1705.03122.pdf) </br>
[9] [Identifying Well-formed Natural Language Questions](https://arxiv.org/pdf/1808.09419.pdf) <br>
