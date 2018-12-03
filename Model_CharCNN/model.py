# model.py

import torch
from torch import nn
import numpy as np
from utils import *

class CharCNN(nn.Module):
    def __init__(self, config, vocab_size, embeddings):
        super(CharCNN, self).__init__()
        self.config = config
        embed_size = vocab_size
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)
        
        # This stackoverflow thread explains how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_size, out_channels=self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        ) # (batch_size, num_channels, (seq_len-6)/3)
        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        ) # (batch_size, num_channels, (seq_len-6-18)/(3*3))
        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        ) # (batch_size, num_channels, (seq_len-6-18-18)/(3*3))
        conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        ) # (batch_size, num_channels, (seq_len-6-18-18-18)/(3*3))
        conv5 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        ) # (batch_size, num_channels, (seq_len-6-18-18-18-18)/(3*3))
        conv6 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        ) # (batch_size, num_channels, (seq_len-6-18-18-18-18-18)/(3*3*3))
        
        # Length of output after conv6        
        conv_output_size = self.config.num_channels * ((self.config.seq_len - 96) // 27)
        
        linear1 = nn.Sequential(
            nn.Linear(conv_output_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear2 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear3 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.output_size),
            nn.Softmax()
        )
        
        self.convolutional_layers = nn.Sequential(conv1,conv2,conv3,conv4,conv5,conv6)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)
    
    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1,2,0) # shape=(batch_size,embed_size,seq_len)
        conv_out = self.convolutional_layers(embedded_sent)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        linear_output = self.linear_layers(conv_out)
        return linear_output
    
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
        if (epoch > 0) and (epoch % 3 == 0):
            self.reduce_lr()
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label).type(torch.LongTensor)
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