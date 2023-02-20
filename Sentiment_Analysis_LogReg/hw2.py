import re
import sys
from typing import List

import nltk
from nltk.tag import pos_tag
import numpy
from numpy.core.defchararray import index
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):# -> list[tuple(list(str), int)]:
    #pass
    f = open(corpus_path, 'r')
    reviews = f.read().split('\n')
    liTup = []
    for rev in reviews:
        spl = rev.split('\t')
        if (len(spl)==2):
            words = spl[0].split(' ')
            liTup.append(tuple([words, int(spl[1])]))
    return(liTup)


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):# -> bool:
    #pass
    if word in negation_words:
        return True
    elif word.endswith("n't"):
        return True
    else:
        return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    #pass
    #posTags = nltk.pos_tag(' '.join(snippet))
    posTags = nltk.pos_tag(snippet)
    negTagList = []
    negTag = False
    for ind, word in enumerate(snippet):
        appended = False
        if not negTag:
            negTagList.append(word)
            appended = True
        isNeg = is_negation(word)
        if isNeg:
            if (ind+1 < len(snippet)) and not (snippet[ind+1] == 'only') :
                negTag = True
        if word in sentence_enders or word in negation_enders:
            negTag = False
        comp = posTags[ind][1]
        if comp == 'JJR' or comp == 'RBR':
            negTag = False
        if not appended:
            if negTag:
                negTagList.append(('NOT_'+word))
            else:
                negTagList.append(word)
    return (negTagList)
        



# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    #pass
    feature_dict = {}
    posnCtr = 0
    for tup in corpus:
        snippet = tup[0]
        for word in snippet:
            if word not in feature_dict:
                feature_dict[word] = posnCtr
                posnCtr+=1
    return (feature_dict)
    

# Converts a snippet into a feature vector
# snippet is a list of strings
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    #pass
    featureVector = numpy.zeros(len(feature_dict))
    for word in snippet:
        if word in feature_dict:
            index = feature_dict[word]
            featureVector[index]+=1
    return (featureVector)


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    #pass
    n = len(corpus)
    d = len(feature_dict)
    X = numpy.empty([n, d])
    Y = numpy.empty(n)
    for index, tup in enumerate(corpus):
        snippet = tup[0]
        X[index] = vectorize_snippet(snippet, feature_dict)
        Y[index] = int(tup[1])
    return (tuple([X,Y]))


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    #pass
    rows = len(X)
    columns = len(X[0])
    for j in range(columns):
        min = max = X[0][j]
#        print('Initial min, max: ', min, max)
        for featureVector in X:
            f = featureVector[j]
#            print('f while setting min-max: ', f)
            if max<f:
                max=f
            if min>f:
                min=f
#        print('Set min, max: ', min, max)
        for featureVector in X:
            f = featureVector[j]
#            print('f during replacements: ', f)
            if min==max:
                featureVector[j] = 0
#                print('direct zero')
            else:
                featureVector[j] = (f - min)/(max - min)
#                print('f, min, max: ', f, min, max)
#            print('Normalized f: ', featureVector[j])
#    for feature in X:
#        max = min = feature[0]
#        for f in feature:
#            if max<f:
#                max=f
#            if min>f:
#                min=f
#        for i,f in enumerate(feature):
#            if min==max:
#                feature[i] = 0
#            else:
#                feature[i] = (f - min)/(max - min)
    


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    #pass
    corpus = load_corpus(corpus_path)
    for snippet in corpus:
        snippet = tuple([tag_negation(snippet[0]), snippet[1]])
    featureDict = get_feature_dictionary(corpus)
    X, Y = vectorize_corpus(corpus, featureDict)
    normalize(X)
    model = LogisticRegression()
    model.fit(X, Y)
    return (model, featureDict)



# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    #pass
    tp = fp = fn = 0
    precision = recall = fmeasure = 0
    for i in range(len(Y_pred)):
        tl = Y_test[i]
        pl = Y_pred[i]
        if tl==1 and pl==1:
            tp+=1
        if tl==0 and pl==1:
            fp+=1
        if tl==1 and pl==0:
            fn+=1
    if not (tp+fp==0):
        precision = tp/(tp+fp)
    if not (tp+fn==0):
        recall = tp/(tp+fn)
    if not (precision+recall==0):
        fmeasure = 2*precision*recall/(precision+recall)
    return (precision, recall, fmeasure)


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    #pass
    testCorpus = load_corpus(corpus_path)
    for snippet in testCorpus:
        snippet = tuple([tag_negation(snippet[0]), snippet[1]])
    X, Y_test = vectorize_corpus(testCorpus, feature_dict)
    normalize(X)
    Y_pred = model.predict(X)
    return (evaluate_predictions(Y_pred, Y_test))


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    #pass
    coef = logreg_model.coef_
    #print(coef)
    liTup = []
    for index in range(len(coef[0])):
        liTup.append(tuple([index, coef[0][index]]))
    liTup.sort(key= lambda x: abs(x[1]), reverse=True)
    #print(liTup)
    unigramList = list(feature_dict)
    for ind, tup in enumerate(liTup):
        liTup[ind] = tuple([unigramList[tup[0]], tup[1]])
    #print (liTup)
    return (liTup[:k])


def main(args):
    model, feature_dict = train('train.txt')
    #print('Model: ', model)
    #print('Feature Dictionary: ', feature_dict)
    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict, 5)
    for weight in weights:
        print(weight)

    #load_corpus('test.txt')
    #tag_negation(['hello', 'there', '!', 'are', 'you', 'not', 'general', 'kenobi', '?'])
    #X = numpy.array([[1.0,2.0,3.0,4.0,5.0],[6.0,8.0,10.0,7.0,9.0],[11.0,12.0,13.0,14.0,15.0],[17.0,1.0,3.0,24.0,10.0]])
    #normalize(X)
    #print(X)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
