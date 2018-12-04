# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 1
    hidden_size = 64
    output_size = 4
    max_epochs = 15
    hidden_size_linear = 64
    lr = 0.5
    batch_size = 128
    seq_len = None # Sequence length for RNN
    dropout_keep = 0.8
