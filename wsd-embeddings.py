"""
wsd-embeddings.py
Ricardo Garriga-Ramos
CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - March 9 Spring 2024



Discription: 
Use glove second order co-occurence vectors by to tokenize training and testing documents
train an SVM or Neural Network inorder to disembiguate the sense of the word line


In thesting there is a clear imporvment from the use of glove embedings
This makes sense especally in SVM where splitting space is the central premis of the algorithm
As for my Neural network I used a 100 epoch .01 learn rate and 3 hiden layer as spesified in the class constructor
These peramiters gave me the best result but still underproformed SVM which is unsadisfacory given that it is a more complex algorithm

Confusion Matrices were phone is positive and product is Negitive
assignment 5
SVM
True positive = 69      True Negitive = 52
False Positive = 2      False Negitive = 3
Accuracy = 0.9603174603174603

Neural Network
True positive = 64      True Negitive = 53
False Positive = 1      False Negitive = 8
Accuracy = 0.9285714285714286


assignment 4
GaussianNaiveBayes
True positive = 62      True Negitive = 45
False Positive = 9      False Negitive = 10
Accuracy = 0.8492063492063492

RandomForestClassifier
True positive = 65      True Negitive = 49
False Positive = 5      False Negitive = 7
Accuracy = 0.9047619047619048

SVM
True positive = 65      True Negitive = 45
False Positive = 9      False Negitive = 7
Accuracy = 0.873015873015873


assignment 3
Discussion list
True positive = 55      True Negitive = 49
False Positive = 5      False Negitive = 17
Accuracy = 0.8253968253

Most frequent sense baseline
Accuracy = 0.4285714286


python3 wsd-embeddings.py line-train.txt line-test.txt SVM > my-line-answers.txt
python3 wsd-embeddings.py line-train.txt line-test.txt NN > my-line-answers.txt
or
python3 wsd-embeddings.py line-train.txt line-test.txt  > my-line-answers.txt
which defults to SVM
"""

import sys
import re
import string
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction import text

import torch
from torch import nn
from torch.nn import functional as F

# run time
import time
start_time = time.time()



# global variables for glove embedings as dictionary

# glove.6B/glove.6B.300d.txt
filepath = 'glove.6B/glove.6B.50d.txt'
glove_embedings = {}
file = open(filepath, encoding="utf8")
for line in file:
    word, *vector = line.split()
    glove_embedings[word]= vector
# --- 8.484450340270996 seconds --- on d50
# print("--- %s seconds ---" % (time.time() - start_time))


def vectorize_training(stop_list):
    # intialize vectorization
    vectors_dict = {'senseid':[]} 
    embeding_profiles = []

    # for each line in the training set find the context and senseid
    has_context = False
    while True:
        cur_line = train_file.readline()
        if not cur_line:
            break
        # vectorize line
        if has_context:
            embeding_profiles.append(parse_context(stop_list, cur_line.lower().split(" ")))
            has_context = False
        # detect line
        elif re.search("<context>", cur_line):
            has_context = True
        # get senseid
        elif re.search("<answer instance=", cur_line):
            senseid = cur_line.split('"')[3]
            vectors_dict['senseid'].append(senseid)
    
    # turn word embedings to vectors
    context_to_embedings(vectors_dict, embeding_profiles)
    return vectors_dict
def vectorize_test(stop_list):
    # intialize vectorization
    vectors_dict = {'instanceid':[]} 
    embeding_profiles = []
   
    # for each line in the training set find the context and instanceid
    has_context = False
    while True:
        cur_line = test_file.readline()
        if not cur_line:
            break
        # vectorize line
        if has_context:
            embeding_profiles.append(parse_context(stop_list, cur_line.lower().split(" ")))
            has_context = False
        # detect line
        elif re.search("<context>", cur_line):
            has_context = True
        # get instance
        elif re.search("<instance id=", cur_line):
            instanceid = cur_line.split('"')[1]
            vectors_dict['instanceid'].append(instanceid)

    # turn word embedings to vectors
    context_to_embedings(vectors_dict, embeding_profiles)
    return vectors_dict




def parse_context(stop_list, words):
    # create a profile for the line
    embedding_words = []
    count = 0

    for word in words:
        is_head = re.search("<head>(lines?)</head>", word)
        if is_head:
            # is key word
            # not in use
            key = is_head.group(1)

        elif not re.search("<s>|</s>|<p>|</p>|<@>", word):

            # remove stop words
            if word not in stop_list:
                word = word.translate(str.maketrans('', '', string.punctuation))
                if word not in stop_list:
                    
                    
                    if re.search("[0-9]+", word):
                        # is a number
                        # not in use
                        num = "<numaric>"
                    else:
                        # is a word
                        count += 1
                        embedding_words.append(word)
    #add vector profile to dictioinary
    return embedding_words
    

def context_to_embedings(vectors_dict, embeding_profiles):
    # initalize dimentions of vectors 
    for column in range(len(glove_embedings['the'])):
        vectors_dict[column] = []
        for profile in embeding_profiles:
            vectors_dict[column].append(0)

    # sum all dense vectors from relevent words as the vector for the line
    for row,profile in enumerate(embeding_profiles):
        for word in profile:
            if word in glove_embedings.keys():
                for column in range(len(glove_embedings['the'])):
                    vectors_dict[column][row] += float(glove_embedings[word][column])
        
        # take the avrage of the sum
        for column in range(len(glove_embedings['the'])):
            vectors_dict[column][row] = vectors_dict[column][row]/len(profile)

    



def train(algorithm, train_vectors_df):
    # train an return model based on algorithm
    
    if algorithm == 'NN':
        return train_nn(train_vectors_df)
    elif algorithm == 'SVM':
        X = train_vectors_df.drop('senseid', axis = 1)
        y = train_vectors_df['senseid']
        clf = svm.SVC()
        return clf.fit(X, y)    

def test(model, test_vectors_df, algorithm):
    # find instanceids
    inst_list = test_vectors_df['instanceid']

    # find predicted senseids
    if algorithm == 'NN':
        lable_list = test_nn(test_vectors_df, model)
    elif algorithm == 'SVM':
        lable_list = model.predict(test_vectors_df.drop('instanceid', axis = 1))

    # print to myline answers
    for i in range(len(inst_list)):
        print(f"<answer instance=\"{inst_list[i]}\" senseid=\"{lable_list[i]}\"/>")




def train_nn(train_vectors_df):
    # create Neural Network from class
    model = NeuralNetwork()


    X = train_vectors_df.drop('senseid', axis = 1).to_numpy()
    y = pd.factorize(train_vectors_df['senseid'])
    # df to tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y[0], dtype=torch.float32).reshape(-1, 1)

    # we want to maximize accuracy
    criterion = nn.L1Loss()
    #Optimizer, learning rate and epochs
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    epochs = 100
    losses = []
    for i in range(epochs):
        # walk through the neural network and learn from the data
        y_pred = model.forward(X)
        loss = criterion(y_pred, y)

        # used for debuging
        """losses.append(loss.detach().numpy())
        if i% 10 == 0:
            print(y_pred,y)
            print(loss)"""

        # back propigation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return model




def test_nn(test_vectors_df, model):
    X = test_vectors_df.drop('instanceid', axis = 1).to_numpy()
    # df to tensor
    X = torch.tensor(X, dtype=torch.float32)

    # aply the test data set to the model
    with torch.no_grad():
        y_out = model.forward(X)


    # turn y_out from floating point numbers to strings
    lable_list = []
    for i in y_out:
        if(round(i[0].item()) == 0):
            lable_list.append('phone')
        else:
            lable_list.append('product')

    return lable_list


def main():
    # use stoplist from skikit
    my_words = ['','\n']
    stop_list = text.ENGLISH_STOP_WORDS.union(my_words)

    # Creeate vector representations for the training and testing data as a dictionary and dataframe
    train_vectors_dict = vectorize_training(stop_list)
    test_vectors_dict = vectorize_test(stop_list)
  
    train_vectors_df = pd.DataFrame.from_dict(train_vectors_dict)
    test_vectors_df = pd.DataFrame.from_dict(test_vectors_dict)
    
    
    # Train and test the algorithm on the data
    model = train(algorithm, train_vectors_df)
    test(model, test_vectors_df, algorithm)





class NeuralNetwork(nn.Module):
    #50 inputs -> hiden layers(nurons) -> output (2 classes)
    def __init__ (self, in_features = 50, h1=120,h2=140,h3=130, out_features = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,h3)
        self.out = nn.Linear(h3, out_features)

    def forward (self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x













# parse arguments and start program
if len(sys.argv) >= 3:
    # open files
    train_file = open(sys.argv[1], 'r')
    test_file = open(sys.argv[2], 'r')

    # spesified l-model  len(sys.argv) = 4
    if len(sys.argv) == 4:
        algorithm = sys.argv[3]
    # unspesified l-model  len(sys.argv) = 3
    elif len(sys.argv) == 3:
        algorithm = 'SVM'
    
    """if algorithm =='NN':
        # check what hardware is avalable for neural network
        device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )"""


    main()

    train_file.close()
    test_file.close()
else:
    print('Not enough arguments')
