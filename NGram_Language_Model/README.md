# N-Gram Language Model with Laplace Smoothing
#### Make sure you have downloaded the data for this assignment:
* warpeace.txt, Tolstoy’s War and Peace
* shakespeare.txt, the complete plays of Shakespeare
#### Make sure you have installed the following libraries:
* NLTK, https://www.nltk.org/
* NLTK corpora, https://www.nltk.org/data.htm
## 1. Basic N-Gram Language Model
### Function get_ngrams(n, text)
* Pad text with enough start tokens ‘< s >’ so that you’re able to make n-grams for the beginning of the sentence, plus a single end token ‘< /s >’, which we will need later in Part 3. 
* For each “real,” non-start token, yield an n-gram tuple of the form (word, context), where word is a string and context is a tuple of the n − 1 preceding words/strings.
### Class NGramLM
* Keep track of counts from the training data. Look over the initialization method.
* The initialization method saves n as an internal variable self.n and initializes three other internal variables: 
  - self.ngram counts, a dictionary for counting n-grams seen in the training data, 
  - self.context counts, a dictionary for counting contexts seen in the training data, 
  - self.vocabulary, a set for keeping track of words seen in the training data.
### Function update(self, text)
* Use get ngrams(n, text) to get n-grams of the appropriate size for this NGramLM. 
* For each n-gram, update the internal counts as needed. (Think about how we should handle the start and end tokens, ‘< s >’ and ‘< /s >’) 
  - The keys of ngram counts should be tuples of the form (word, context) where context is a tuple. 
  - The keys of context counts should be tuples of strings.
### Function load_corpus(corpus_path)
* Open the file at corpus path and load the text. 
* Tokenize the text into sentences. 
  – Split the text into paragraphs. The corpus file has blank lines separating paragraphs. 
  – Use the NLTK function sent tokenize() to split the paragraphs into sentences. 
* Use the NLTK function word tokenize() to split each sentence into words. 
* Return a list of all sentences in the corpus, where each sentence is a list of words.
### Function create_ngramlm(n, corpus_path)
* Use load corpus to load the data from corpus path. 
* Create a new NGramLM and use its update() function to add each sentence from the loaded corpus. 
* Return the trained NGramLM.
### Function get_ngram_prob(self, word, context, delta=0)
* Use the counts stored in the internal variables to calculate and return pMLE(word|context). 
* If context is previously unseen (ie. not in the training data), return 1/|V |, where V is the model’s vocabulary.
### Function get_sent_log_prob(self, sent, delta=0)
To predict the probability of a sentence, we multiply together its individual n-gram probabilities. This can be a very small number, so to avoid underflow, we will report the sentence’s log probability instead.
* Use get ngrams() to get the n-grams in sent. 
* For each n-gram, use get ngram prob() to get the n-gram probability and take the base-2 logarithm using math.log(). 
* Return the sum of the n-gram log probabilities.

## 2. Smoothing
### Update NGramLM.get_ngram_prob ()
* We need to add support for out-of-vocabulary words. Let’s implement Laplace smoothing. 
* Update NGramLM.get ngram prob () to support Laplace-smoothed probabilities using the delta argument. 
* Check if delta is 0. If so, it should return the same probability as it would have before. 
* If delta is not 0, apply Laplace smoothing and return the smoothed probability. 

## 3. Evaluation
### Function NGramLM.get_perplexity(self, corpus)
* Use NGramLM.get sent log prob() to get corpus-level log probability. 
* Divide by the total number of tokens in the corpus to get the average log probability. 
* Use math.pow() to calculate the perplexity.
### Function NGramLM.generate_random_word(self, context, delta=0)
* Sort self.vocabulary according to Python’s default ordering (basically alphabetically order). 
* Generate a random number (0.0 <= r < 1.0) using random.random(). This value r is how you know which word to return. 
* Iterate through the words in the sorted vocabulary and use NGramLM.get ngram prob() to get their probabilities given context. Make sure to pass delta. 
* These probabilities all sum to 1.0, so if we imagine aNGramLM.generate random text(self, max length, delta=0) number line from 0.0 to 1.0, the space on that number line can be divided up into zones corresponding to the words in the vocabulary. For example, if the first words are “apple” and “banana,” with probabilities 0.09 and 0.57, respectively, then 0.0 <= r < 0.9 belongs to “apple” and 0.9 <= r < 0.66 belongs to “banana,” and so on. Return the word whose zone contains r.
### Function NGramLM.generate_random_text(self, max_length, delta=0)
* Generate the first word using NGramLM.generate random word() with a context consisting of start tokens ‘< s >’. 
* Continue generating using the previously generated words as context for each new word. 
* Stop generating when either max length is reached, or if the stop token ‘< /s >’ is generated. 
* Return the generated sentence as a single string. 

#### We are all set! You can modify main() and use the provided data files warpeace.txt and shakespeare.txt to test your code and make sure it gives reasonable outputs.
