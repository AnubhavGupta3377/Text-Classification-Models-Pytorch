# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings,PositionalEncoding
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np
from utils import *

class Transformer(nn.Module):
    def __init__(self, config, src_vocab):
        super(Transformer, self).__init__()
        self.config = config
        
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab), deepcopy(position)) #Embeddings followed by PE

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )
        
        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sents = self.src_embed(x.permute(1,0)) # shape = (batch_size, sen_len, d_model)
        encoded_sents = self.encoder(embedded_sents)
        
        # Convert input to (batch_size, d_model) for linear layer
        final_feature_map = encoded_sents[:,-1,:]
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
    
            if i % 100 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
                
        return train_losses, val_accuracies