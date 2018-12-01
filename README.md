# Text-Classification-Models
Implementation of State-of-the-art Text Classification Models in Pytorch

## Current Model Implementations
- **fastText:** fastText Model from [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
- **TextCNN:** CNN for text classification proposed in [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- **TextRNN:** Bi-direction LSTM network for text classification
- **Seq2seq With Attention:** Implementation of seq2seq model with attention from [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), [Text Classification Research with Attention-based Recurrent Neural Networks](http://univagora.ro/jour/index.php/ijccc/article/download/3142/pdf)

## Upcoming Models
- **RCNN:** Implementation of RCNN Model proposed in [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
- **Hierarchical Attention:** Implementation of hierarchical attention model for text classification as proposed in [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-uys7{border-color:inherit;text-align:center}
.tg .tg-abip{background-color:#f9f9f9;border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-btxf{background-color:#f9f9f9;border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-uys7" rowspan="3"><span style="font-weight:bold">Model</span></th>
    <th class="tg-c3ow" colspan="4"><span style="font-weight:bold">Dataset</span></th>
  </tr>
  <tr>
    <td class="tg-abip" colspan="2"><span style="font-weight:bold">AG_News</span></td>
    <td class="tg-abip" colspan="2"><span style="font-weight:bold">Query_Well_formedness</span></td>
  </tr>
  <tr>
    <td class="tg-abip"><span style="font-weight:bold">Accuracy (%)</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Runtime</span></td>
    <td class="tg-abip"><span style="font-weight:bold">Accuracy (%)</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Runtime</span></td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">fastText</span></td>
    <td class="tg-btxf">89.46</td>
    <td class="tg-0pky">16.0 Mins</td>
    <td class="tg-btxf"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">TextCNN</span></td>
    <td class="tg-btxf">88.57</td>
    <td class="tg-0pky">17.2 Mins</td>
    <td class="tg-btxf"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">TextRNN</span><br></td>
    <td class="tg-btxf">88.07 (Sequence length = 20)<br><br>90.43 (Flexible sequence length)<br></td>
    <td class="tg-0pky">21.5 Mins<br><br>36.8 Mins<br></td>
    <td class="tg-btxf"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">Seq2Seq_Attention</span></td>
    <td class="tg-btxf">90.26</td>
    <td class="tg-0pky">19.10 Mins</td>
    <td class="tg-btxf"></td>
    <td class="tg-0pky"></td>
  </tr>
</table>

<table>
  <tr>
    <td rowspan="3">Model</td>
    <td align="center" colspan="4">Dataset</td>
  </tr>
  <tr>
    <td colspan="2">AG_News</td>
    <td colspan="2">Query_Well_formedness</td>
  </tr>
  <tr>
    <td>Accuracy </td>
    <td>Runtime </td>
    <td>Accuracy </td>
    <td>Runtime </td>
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
    <td>Seq2Seq_Attention</td>
    <td>90.26</td>
    <td>19.10 Mins</td>
    <td></td>
    <td></td>
  </tr>
</table>
