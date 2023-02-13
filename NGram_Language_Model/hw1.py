import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    numOfWords = len(text)
    modifiedText = ['<s>' for i in range(1, n)]
    modifiedText.extend(text)
    modifiedText.append('</s>')
    for j in range(numOfWords+1):
        string = modifiedText[j+n-1]
        context = tuple(modifiedText[j:j+n-1])
        yield (string, context)



# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    text = []
    with open(corpus_path, 'r') as fp:
        text = fp.read().split('\n\n')
    corpusTokens = []
    for paragraph in text:
        corpusTokens.extend(word_tokenize(sent) for sent in sent_tokenize(paragraph))
    return corpusTokens


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    nGramLangModel = NGramLM(n)
    for data in load_corpus(corpus_path):
        nGramLangModel.update(data)
    return nGramLangModel


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        for ng in get_ngrams(self.n, text):
            self.ngram_counts[ng] = self.ngram_counts.get(ng, 0) + 1
            self.vocabulary.add(ng[0])
            self.context_counts[ng[1]] = self.context_counts.get(ng[1], 0) + 1

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        vocabSize = len(self.vocabulary)
        if vocabSize==0:
            return 0
        if context in self.context_counts:
            ng = tuple([word, context])
            return (self.ngram_counts.get(ng,0)+delta)/(self.context_counts[context]+(delta*vocabSize))
        return 1.0/vocabSize

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        ret = 0.0
        for word, context in get_ngrams(self.n, sent):
            try:
                ret += math.log2(self.get_ngram_prob(word, context, delta))
            except ValueError:
                ret -= math.inf
            except:
                print('Some error apart from ValueError')
        return ret

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]], delta=.0) -> float:
        # numOfTokens = sum([len(sentences) for sentences in corpus])
        # numOfTokens = 0
        wordsList = list()
        for sentence in corpus:
            for word in sentence:
                wordsList.append(word)
        corpusLgProb = self.get_sent_log_prob(word_tokenize(' '.join(wordsList)), delta)
        return math.pow(2, -corpusLgProb/len(wordsList))

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        vocabAlpha = sorted(self.vocabulary)
        rand = random.random()
        l = r = 0
        for w in vocabAlpha:
            l, r = r, r+self.get_ngram_prob(w, context, delta)
            if l<=rand and rand<r:
                return w

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        text = []
        ctx = ['<s>']*(self.n-1)
        for i in range(max_length):
            genW = self.generate_random_word(tuple(ctx),delta)
            text.append(genW)
            if genW=='</s>':
                break
            ctx.append(genW)
            ctx = ctx[1:]
        return ' '.join(text)
                


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(3, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'

    print(trigram_lm.get_sent_log_prob(word_tokenize(s1)))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2)))

    test_path = 'shakespeare.txt'
    uni = create_ngram_lm(1, test_path)
    tri = create_ngram_lm(3, test_path)
    penta = create_ngram_lm(5, test_path)
    print("\n***GENERATED UNIGRAM SENTENCES***")
    for i in range(5):
        print('Sent', (i+1), ':', uni.generate_random_text(10, 0.01))
    print("\n***GENERATED TRIGRAM SENTENCES***")
    for i in range(5):
        print('Sent', (i+1), ':', tri.generate_random_text(10, 0.03))
    print("\n***GENERATED 5-GRAM SENTENCES***")
    for i in range(5):
        print('Sent', (i+1), ':', penta.generate_random_text(10, 0.05))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
