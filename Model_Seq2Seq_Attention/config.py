# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 1
    hidden_size = 32
    bidirectional = True
    output_size = 4
    max_epochs = 15
    lr = 0.5
    batch_size = 128
    dropout_keep = 0.8
    max_sen_len = None # Sequence length for RNN