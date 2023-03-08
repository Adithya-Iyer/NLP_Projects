from random import randint, random
import sys

import nltk
from nltk.corpus import brown
import numpy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    return (corpus_sents, corpus_tags)


# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings
def get_ngram_features(words, i):
    arg = ['<s>','<s>']+list(words)+['</s>','</s>']
    i += 2
    pb = 'prevbigram'+'-'+arg[i-1]
    nb = 'nextbigram'+'-'+arg[i+1]
    ps = 'prevskip'+'-'+arg[i-2]
    ns = 'nextskip'+'-'+arg[i+2]
    pt = 'prevtrigram'+'-'+arg[i-1]+'-'+arg[i-2]
    nt = 'nexttrigram'+'-'+arg[i+1]+'-'+arg[i+2]
    ct = 'centertrigram'+'-'+arg[i-1]+'-'+arg[i+1]
    return [pb, nb, ps, ns, pt, nt, ct]


# Generate word-based features
# word is a string
# returns a list of strings
def get_word_features(word):
    wrd = 'word-'+word
    wbf = [wrd]
    if word[0].isupper():
        wbf.append('capital')
        if word.isupper():
            wbf.append('allcaps')
    ws = ['wordshape-']
    sws = ['short-wordshape-']
    hasnum = False
    hashyp = False
    for ch in word:
        if ch.isupper():
            ws.append('X')
            if sws[-1]!='X':
                sws.append('X')
        elif ch.islower():
            ws.append('x')
            if sws[-1]!='x':
                sws.append('x')
        elif ch.isdigit():
            hasnum = True
            ws.append('d')
            if sws[-1]!='d':
                sws.append('d')
        else:
            ws.append(ch)
            sws.append(ch)
        if ch=='-':
            hashyp = True
    wbf.append(''.join(ws))
    wbf.append(''.join(sws))
    if hasnum:
        wbf.append('number')
    if hashyp:
        wbf.append('hyphen')
    prefixes, suffixes = [], []
    for i in range(min(4,len(word))):
        j = i+1
        pre = 'prefix'+str(j)+'-'+word[:j]
        prefixes.append(pre)
        suf = 'suffix'+str(j)+'-'+word[-j:]
        suffixes.append(suf)
    wbf.extend(prefixes)
    wbf.extend(suffixes)
    return wbf


# Wrapper function for get_ngram_features and get_word_features
# words is a list of strings
# i is an int
# prevtag is a string
# Returns a list of strings
def get_features(words, i, prevtag):
    feats = get_ngram_features(words, i) + get_word_features(words[i])
    bigramFt = 'tagbigram-'+prevtag
    feats.append(bigramFt)
    ftList = []
    for ft in feats:
        if ft.startswith('wordshape-') or ft.startswith('short-wordshape-'):
            ftList.append(ft)
        else:
            ftList.append(ft.lower())
    return ftList


# Remove features that occur fewer than a given threshold number of time
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# threshold is an int
# Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
    featCnt = {}
    for words in corpus_features:
        for w in words:
            for f in w:
                featCnt[f] = featCnt.get(f,0) + 1
    cfv2 = []
    rareFt = set()
    commonFt = set()
    for j in range(len(corpus_features)):
        words = corpus_features[j]
        cfv2.append([])
        for i in range(len(words)):
            w = words[i]
            cfv2[j].append([])
            for f in w:
                if featCnt[f]<threshold:
                    rareFt.add(f)
                else:
                    commonFt.add(f)
                    cfv2[j][i].append(f)
    return (cfv2, commonFt)


# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)
def get_feature_and_label_dictionaries(common_features, corpus_tags):
    featDict, tagDict = {}, {}
    index = 0
    for feat in common_features:
        featDict[feat] = index
        index += 1
    index = 0
    for sent in corpus_tags:
        for wt in sent:
            if wt not in tagDict:
                tagDict[wt] = index
                index += 1
        if index==12:
            break
    return (featDict, tagDict)

# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    tag_list = []
    for stags in corpus_tags:
        for wtags in stags:
            tag_list.append(tag_dict[wtags])
    return numpy.array(tag_list)

# Build a sparse input matrix X
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# feature_dict is a dictionary {string: int}
# Returns a Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
    rows, cols = [], []
    index = 0
    for sent in corpus_features:
        for wrd in sent:
            for feat in wrd:
                rows.append(index)
                if feat in feature_dict:
                    cols.append(feature_dict[feat])
                else:
                    cols.append(randint(0, len(feature_dict)-1))
            index += 1
    values = [1 for i in range(len(cols))]
    rows, cols, values = numpy.array(rows), numpy.array(cols), numpy.array(values)
    return csr_matrix((values, (rows, cols)), (index, len(feature_dict)))


# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    csents, ctags = load_training_corpus(proportion)
    cfeats = []
    for j, sent in enumerate(csents):
        cfeats.append([])
        for i in range(len(sent)):
            prevtag = '<s>' if i==0 else ctags[j][i-1]
            cfeats[j].append(get_features(sent, i, prevtag))
    cfv2, comfeats = remove_rare_features(cfeats)
    ftDict, tagDict = get_feature_and_label_dictionaries(comfeats, ctags)
    X, y = build_X(cfv2, ftDict), build_Y(ctags, tagDict)
    logreg = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial')
    logreg.fit(X, y)
    return (logreg, ftDict, tagDict)



# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)
def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
    words = test_sent[0]
    n = len(words)
    T = len(reverse_tag_dict)
    Y_pred = numpy.empty((n-1, T, T))
    for i in range(1, n):
        features = []
        for ind in range(T):
            tprime = reverse_tag_dict[ind]
            ftList = get_features(words, i, tprime)
            features.append(ftList)
        X_test = build_X([features], feature_dict)
        logprob = model.predict_log_proba(X_test)
        Y_pred[i-1] = logprob
    feat0 = get_features(words, 0, '<s>')
    X_start = build_X([[feat0]], feature_dict)
    Y_start = model.predict_log_proba(X_start)
    # Y_start = numpy.array(Y_start).reshape(1,-1)
    return (Y_start, Y_pred)


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    yshape = Y_pred.shape
    n, T = yshape[0]+1, yshape[1]
    V = numpy.empty((n, T))
    BP = numpy.empty((n, T))
    V[0] = Y_start
    for i in range(1,n):
        for j in range(T):
            dp = numpy.empty(T)
            for ind in range(T):
                dp[ind] = V[i-1][ind] + Y_pred[i-1][ind][j]
            V[i][j] = numpy.max(dp)
            BP[i][j] = int(numpy.argmax(dp))
    tagInd = []
    bpt = int(numpy.argmax(V[n-1]))
    for x in range(n):
        tagInd.append(bpt)
        posn = n-(x+1)
        bpt = int(BP[posn][bpt])
    return list(reversed(tagInd))




# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    test_corpus = load_test_corpus(corpus_path)
    reverse_tag_dict = {tag_dict[k]:k for k in tag_dict}
    pred = []
    for line in test_corpus:
        Y_start, Y_pred = get_predictions(line, model, feature_dict, reverse_tag_dict)
        tagInd = viterbi(Y_start, Y_pred)
        tagSeq = []
        for ind in tagInd:
            tag = reverse_tag_dict[ind]
            tagSeq.append(tag)
        pred.append(tagSeq)
    return pred


def main(args):
    model, feature_dict, tag_dict = train(0.2)

    predictions = predict('test.txt', model, feature_dict, tag_dict)
    for test_sent in predictions:
        print(test_sent)
    print(get_ngram_features(['the','happy','cat'],0))
    print(get_ngram_features(['the','happy','cat'],1))
    print(get_ngram_features(['the','happy','cat'],2))
    print(get_word_features('UTDallas'))
    print(get_word_features('UTD883-West'))
    print(get_features(['the', 'Happy'], 1, 'DT'))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
