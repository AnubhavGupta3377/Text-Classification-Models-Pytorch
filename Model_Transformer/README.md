# Attention Is All You Need (Transformer)

Transformer was originally proposed for the task of machine translation in NIPS-2017 paper "Attention Is All You Need".

Here, I have implemented the Transformer network for text classification. Some part of the code is based on the [blog post](http://nlp.seas.harvard.edu/2018/04/03/attention.html) by Alexander Rush on Transformer. We have made some modificaitons to make the Transformer work for text classification problem, details of which are given in section "Using Transformer for Classification".

## Transformer Motivation
- Despite GRUs and LSTMs, RNNs still need attention mechanism to deal with long range dependencies. This is because path length for co-dependent computation between stages grow with sequence
- Why not use *just* the attention without the RNNs

## Transformer Overview
- Sequence-to-sequence model
- Encoder-decoder architecture
- Machine translation with parallel corpus
- Predict each translated word
- Final cost function is standard Cross Entropy error on top of a softmax classifier

![Model Architecture of Transformer](images/transformer_overview.png)

## Basic Building Block - Dot Product Attention
**Inputs:** </br>
- A query *q*
- Set of key-value (*k-v*) pairs to an output
- Query, keys, values and output are all vectors

**Output:** </br>
- Weighted sum of values
- Weight of each value is computed using an inner product of query and corresponding key</br>
- Query and keys have dimensionality d<sub>k</sub>
- Values have dimensionality d<sub>v</sub>

![](https://latex.codecogs.com/gif.latex?%24%24%20A%28q%2CK%2CV%29%20%3D%20%5Csum_%7Bi%7D%20%5Cfrac%7Be%5E%7Bq.k_i%7D%7D%7B%5Csum_%7Bj%7D%20e%5E%7Bq.k_j%7D%7D%20v_i%20%24%24)

### Dot-Product Attention - Matrix Notation
- When we have multiple queries *q*, we stack them in a matrix *Q*.

![](https://latex.codecogs.com/gif.latex?%24%24%20A%28Q%2CK%2CV%29%20%3D%20%5Ctext%7Bsoftmax%7D%20%28QK%5ET%29V%20%24%24)

### Scaled Dot-Product Attention
**Problem:** As d<sub>k</sub> gets large, the variance of q<sup>T</sup>k increase</br>
-> some values inside the softmax get large</br>
-> the softmax gets very peaked</br>
-> hence its gradient gets smaller

**Solution:** Scale by length of query/key vectors.

![](https://latex.codecogs.com/gif.latex?%24%24%20A%28Q%2CK%2CV%29%20%3D%20%5Ctext%7Bsoftmax%7D%20%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V%20%24%24)

![](images/scaled_dot_product_attention.png)

## Self-Attention and Multi-head Attention
- Input word vectors could be the queries, keys and values
- The word vectors themselves select each other
- Word vector stacks - Q = K = V
- **Problem:** Only one way for words to interact with each other
- **Solution:** Multi-head attention
- First map Q,K,V into *h* many lower dimensional spaces via **W** matrices
- Then apply attention, then concatenate outputs and pipe through linear layer

![](images/multihead_attention.png)

![](https://latex.codecogs.com/gif.latex?%24%24%20%5Ctext%7BMultihead%7D%28Q%2CK%2CV%29%20%3D%20%5Ctext%7Bconcat%7D%28head_1%2C%20head_2%2C%20%5Ccdots%2C%20head_h%29%20W%5EO%20%24%24)

where

![](https://latex.codecogs.com/gif.latex?%24%24%20head_i%20%3D%20%5Ctext%7BAttention%7D%20%28Q%20W%5EQ_i%2C%20K%20W%5EK_i%2C%20V%20W%5EV_i%29%20%24%24)

- ![](https://latex.codecogs.com/gif.latex?%24%20W%5EQ_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd_%7Bmodel%7D%5Ctimes%20d_%7Bk%7D%7D%20%24%2C%20%24%20W%5EK_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd_%7Bmodel%7D%5Ctimes%20d_%7Bk%7D%7D%20%24%2C%20%24%20W%5EV_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd_%7Bmodel%7D%5Ctimes%20d_%7Bv%7D%7D%20%24%2C%20%24%20W%5EO%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%20d_v%20%5Ctimes%20d_%7Bmodel%7D%7D%20%24)

**Implementation details:**
- *h* = 8
- d<sub>model</sub> = 512
- d<sub>k</sub> = d<sub>v</sub> = d<sub>model</sub> / h = 64

## Encoder Input
- Sentences are encoded using byte-pair encodings
- For the model to make use of the order of the sequence, positional encoding is used

## Complete Encoder
- Encoder is composed of *N*=6 identical layers
- Each layer has 2 sub-layers
(1) Multi-head attention
(2) 2-layer feed-forward network
- Each sublayer also has
(1) Residual (short-circuit)
(2) Layer normalization
- i.e. output of each sublayer is **LayerNorm(*x*+Sublayer(x))**
- In a self-attention layer, all the **keys**, **values** and **queries** come from the same place - the output of previous layer in the encoder.
- Each position in the encoder can attend to all positions in the previous layer of the encoder

## Complete Decoder
- Similar to encoder, decoder is also composed of *N*=6 identical layers
- Each layer of decoder has 3 sub-layers
(1) Maksed multi-head attention over previous decoder outputs
(2) Multi-head attention over output of encoder
(3) 2-layer feed-forward network
- Each sublayer also has
(1) Residual (short-circuit)
(2) Layer normalization
- i.e. output of each sublayer is **LayerNorm(*x*+Sublayer(x))**
- In encoder-decoder attention, **queries** come from previous decoder layer, **keys** and **values** come from output of encoder 
- This allows every position in the decoder to attend over all positions in the input sequence
- Self-attention in the decoder allow each position in decoder to attend to all positions in the decoder *up to and including that position*.
- We need to prevent leftward information flow in the decoder to preserve the auto-regressive property

## Label Smoothing
- Label smoothing is a regularization technique introduced in ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567)
- For classification problems with one correct class, loss functions such as *cross entropy* attain optima by maximizing the log likelihood of the correct class. This may lead to overfitting if the model learns to assign full probability to ground truth label for training examples.
- Label smoothing encourages the model to become less confident about its predictions.
- For a training example with ground-truth label *y*, it replaces the (true) label distribution ![](https://latex.codecogs.com/gif.latex?%24%24q%28k%7Cx%29%20%3D%20%5Cdelta_%7Bk%2Cy%7D%24%24) with 

![](https://latex.codecogs.com/gif.latex?%24%24%20%5Chat%7Bq%7D%28k%7Cx%29%20%3D%20%281-%5Cepsilon%29%5Cdelta_%7Bk%2Cy%7D%20&plus;%20%5Cepsilon%20u%28k%29%20%24%24)

where *u(k)* is a distribution over the labels independent of the training example *x*, such as the "uniform distribution"; and ![](https://latex.codecogs.com/gif.latex?%5Cepsilon) is a smoothing parameter.

- Then, minimizing the cross-entropy loss is equivalent to minimizing the KL-divergence between label distribution and predicted distribution. This is because

![](https://latex.codecogs.com/gif.latex?%24%24%20CE%28%5Chat%7Bq%7D%2Cp%29%20%3D%20KL%28%5Chat%7Bq%7D%7C%7Cp%29%20&plus;%20H%28%5Chat%7Bq%7D%29%20%24%24)

## Using Transformer for Classification
- Only encoder part of Transformer model is used for classification
- No residual connection, no layer normalization
- No need for masking
- Multihead attention and positionwise feedforward network to extract features
- Then, linear layer to get logits

## Implementation Details
- N = 1, i.e. one layer is used
- d_model = 256
- maximum sequence lenght = 60
- Adam optimizer is used with initial learning rate 0.0003. Reducing by half every 1/3 epochs.

## References:
1) [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2) [http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3) [http://web.stanford.edu/class/cs224n/lectures/lecture12.pdf](http://web.stanford.edu/class/cs224n/lectures/lecture12.pdf)
4) [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1705.03122.pdf) (for positional encoding)
