# model.py

import torch
from torch import nn
import numpy as np
from utils import *

class TextCNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(TextCNN, self).__init__()
        self.config = config
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        # Convolutional Layer
        # We use 3 kernels as in original paper: (3,300),(4,300),(5,300)
        self.conv1 = nn.Conv2d(in_channels=self.config.in_channels, out_channels=self.config.num_channels,
                               kernel_size=(self.config.kernel_size[0],self.config.embed_size),
                               stride=1, padding=0)
        self.activation1 = nn.ReLU()
        self.max_out1 = nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0]+1)

        self.conv2 = nn.Conv2d(in_channels=self.config.in_channels, out_channels=self.config.num_channels,
                               kernel_size=(self.config.kernel_size[1],self.config.embed_size),
                               stride=1, padding=0)
        self.activation2 = nn.ReLU()
        self.max_out2 = nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1]+1)
        
        self.conv3 = nn.Conv2d(in_channels=self.config.in_channels, out_channels=self.config.num_channels,
                               kernel_size=(self.config.kernel_size[2],self.config.embed_size),
                               stride=1, padding=0)
        self.activation3 = nn.ReLU()
        self.max_out3 = nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2]+1)
        
        self.dropout = nn.Dropout(self.config.dropout_keep)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(self.config.num_channels*len(self.config.kernel_size), self.config.output_size)
        
        # Softmax non-linearity
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x).permute(1,0,2).unsqueeze(1)
        # embedded_sent.shape = (batch_size=64,1,max_sen_len=20,embed_size=300)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(3) #shape=(64, num_channels, 20-3+1, 1) (squeeze 1)
        activation_out1 = self.activation1(conv_out1)
        max_out1 = self.max_out1(activation_out1).squeeze(2) #shape=(64, num_channels, 1) (squeeze 1)
        
        conv_out2 = self.conv2(embedded_sent).squeeze(3)
        activation_out2 = self.activation2(conv_out2)
        max_out2 = self.max_out2(activation_out2).squeeze(2)
        
        conv_out3 = self.conv3(embedded_sent).squeeze(3)
        activation_out3 = self.activation3(conv_out3)
        max_out3 = self.max_out3(activation_out3).squeeze(2)
        
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        final_feature_map = self.dropout(all_out)
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