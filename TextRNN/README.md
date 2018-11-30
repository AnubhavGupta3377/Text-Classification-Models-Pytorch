# TextRNN (BiLSTM) Model
Here, we have implemented a Bi-directional Long Short Term Memory network in PyTorch.

LSTMs have been very popular for solving text classification problems due to their theoretical property to capture the entire context while representing a sentence.

## Model Architecture
The architecture of Bi-directional LSTM is as follows:

![TextRNN Architecture](images/BiLSTM.JPG)

## Implementation Details
- Used pre-trained Glove Embeddings for initializing word vectors
- 2 Layers of BiLSTM
- Used 32 hidden units within each BiLSTM layer
- Dropout with keep probability 0.8
- Optimizer - Stochastic Gradient Descent
- Loss function - Negative Log Likelihood
- Experimented with flexible sequence lengths and sequences of length 20
