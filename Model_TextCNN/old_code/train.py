# train.py

from utils import *
from config import Config
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import sys
import torch.optim as optim
from torch import nn, Tensor
from torch.autograd import Variable
import torch
from sklearn.metrics import accuracy_score

def get_accuracy(model, test_x, test_y):
    all_preds = []
    test_iterator = data_iterator(test_x, test_y)
    for x, y in test_iterator:
        x = Variable(Tensor(x))
        y_pred = model(x.cuda())
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
    score = accuracy_score(test_y, np.array(all_preds).flatten())
    return score

if __name__=='__main__':
    train_path = '../data/ag_news.train'
    if len(sys.argv) > 2:
        train_path = sys.argv[1]
    test_path = '../data/ag_news.test'
    if len(sys.argv) > 3:
        test_path = sys.argv[2]
        
    train_text, train_labels, vocab = get_data(train_path)
    train_text, val_text, train_label, val_label = train_test_split(train_text, train_labels, test_size=0.2)
    
    # Read Word Embeddings
    w2vfile = '../data/glove.840B.300d.txt'
    word_embeddings = get_word_embeddings(w2vfile, vocab.word_to_index, embedsize=300)
    
    # Get all configuration parameters
    config = Config()
    
    train_x = np.array([encode_text(text, word_embeddings, config.max_sen_len) for text in tqdm(train_text)]) #(num_examples, max_sen_len, embed_size)
    train_y = np.array(train_label) #(num_examples)
        
    val_x = np.array([encode_text(text, word_embeddings, config.max_sen_len) for text in tqdm(val_text)])
    val_y = np.array(val_label)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = CNNText(config)
    model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################

    train_data = [train_x, train_y]
    val_data = [val_x, val_y]

    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_losses,val_accuracies = model.run_epoch(train_data, val_data)
        print("\tAverage training loss: {:.5f}".format(np.mean(train_losses)))
        print("\tAverage Val Accuracy (per 50 iterations): {:.4f}".format(np.mean(val_accuracies)))

        # Reduce learning rate as number of epochs increase
        if i > 0.5 * config.max_epochs:
            print("Reducing LR")
            for g in optimizer.param_groups:
                g['lr'] = 0.1
        if i > 0.75 * config.max_epochs:
            print("Reducing LR")
            for g in optimizer.param_groups:
                g['lr'] = 0.05

    # Get Accuracy of final model
    test_text, test_labels, test_vocab = get_data(test_path)
    test_x = np.array([encode_text(text, word_embeddings, config.max_sen_len) for text in tqdm(test_text)])
    test_y = np.array(test_labels)

    train_acc = get_accuracy(model, train_x, train_y)
    val_acc = get_accuracy(model, val_x, val_y)
    test_acc = get_accuracy(model, test_x, test_y)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))