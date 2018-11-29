# config.py

class Config(object):
    embed_size = 300
    in_channels = 1
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 4
    max_epochs = 10
    lr = 0.25
    batch_size = 64
    max_sen_len = 20
    dropout_keep = 0.6