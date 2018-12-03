# CharCNN Model
This is the implementation of character-level CNNs as proposed in the paper [Character-level Convolutional Networks for Text
Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf).

In CharCNN, input text is represented by a (*l_0*,*d*) matrix. Where *l_0* is the maximum sentence length and d is the dimensionality of character embedding.

Following characters are used for character quantization:

<p>abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{} </p>

## Model Architecture
![CharCNN Architecture](images/CharCNN.png)

Architecture of CharCNN has 9 layers: 6 convolutional layers and 3 fully-connected layers. Input has 70 features, due to above character quantization method and input feature length *l_0* is choosen to be 300 (1014 in the original paper). 2 Dropout layers are inserted between the 3 fully-connected layers.


| Layer | #Output Channels | Kernel | Pool |
|-------|------------------|--------|------|
| 1 | 256 | 7 | 3 |
| 2 | 256 | 3 | 3 |
| 3 | 256 | 3 | N/A |
| 4 | 256 | 3 | N/A |
| 5 | 256 | 3 | N/A |
| 6 | 256 | 7 | 3 |

The shape of output tensor after the last convolution layer is (*l_0* - 96) / 27. Please see comments in **model.py** for details.

The output of final convolutional layer is the input to first-fully connected layer, which has 256 output units (1024 in the original paper). Second-fully connected layer also has 256 output units. Number of output units in final fully-connected layer is determined by the problem.

## Implementation Details (as in original paper)
- ReLU activaion function
- Max-pooling
- Dropout with keep probability 0.5
- Adam optimizer is used
- Initial learning rate is 0.001, halved at every 3 epochs
- Cross Entropy Loss is used

## Learnings from the Implementation
- Negative Log Likelihood Loss doesn't work for CharCNN (Spent 2 days finding the bug in the code, that was not there).
- Character-level CNN is very sensitive to the choice of Optimizer and learning rate. A lot of parameter tuning is required.
- Overall performance is not better, if not worse, than other state of the art text classification methods.
