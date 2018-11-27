# model.py

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score

class CNNText(nn.Module):
    def __init__(self, config):
        super(CNNText, self).__init__()
        self.config = config
        
        # Convolutional Layer
        # We use 3 kernels as in original paper
        # Size of kernels: (3,300),(4,300),(5,300)
        
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
        x = x.unsqueeze(1) # (batch_size,max_seq_len,embed_size) => (batch_size,1,max_seq_len,embed_size)
        
        conv_out1 = self.conv1(x).squeeze(3)
        activation_out1 = self.activation1(conv_out1)
        max_out1 = self.max_out1(activation_out1).squeeze(2)
        
        conv_out2 = self.conv2(x).squeeze(3)
        activation_out2 = self.activation2(conv_out2)
        max_out2 = self.max_out2(activation_out2).squeeze(2)
        
        conv_out3 = self.conv3(x).squeeze(3)
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
    
    def run_epoch(self, train_data, val_data):
        train_x, train_y = train_data[0], train_data[1]
        val_x, val_y = val_data[0], val_data[1]
        iterator = data_iterator(train_x, train_y, self.config.batch_size)
        train_losses = []
        val_accuracies = []
        losses = []
    
        for i, (x,y) in enumerate(iterator):
            self.optimizer.zero_grad()
    
            x = Tensor(x).cuda()
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, torch.cuda.LongTensor(y-1))
            loss.backward()
    
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
    
            if (i + 1) % 50 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                self.eval()
                all_preds = []
                val_iterator = data_iterator(val_x, val_y, self.config.batch_size)
                for j, (x,y) in enumerate(val_iterator):
                    x = Variable(Tensor(x))
                    y_pred = self.__call__(x.cuda())
                    predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
                    all_preds.extend(predicted.numpy())
                score = accuracy_score(val_y, np.array(all_preds).flatten())
                val_accuracies.append(score)
                print("\tVal Accuracy: {:.4f}".format(score))
                self.train()
                
        return train_losses, val_accuracies
