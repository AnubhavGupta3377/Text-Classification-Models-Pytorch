# model.py

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score

class fastText(nn.Module):
    def __init__(self, config):
        super(fastText, self).__init__()
        self.config = config
        
        # Hidden Layer
        self.fc1 = nn.Linear(self.config.embed_size, self.config.hidden_size)
        
        # Output Layer
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.output_size)
        
        # Softmax non-linearity
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        h = self.fc1(x)
        z = self.fc2(h)
        return self.softmax(z)
    
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
                for x, y in val_iterator:
                    x = Variable(Tensor(x))
                    y_pred = self.__call__(x.cuda())
                    predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
                    all_preds.extend(predicted.numpy())
                score = accuracy_score(val_y, np.array(all_preds).flatten())
                val_accuracies.append(score)
                print("\tVal Accuracy: {:.4f}".format(score))
                self.train()
                
        return train_losses, val_accuracies
