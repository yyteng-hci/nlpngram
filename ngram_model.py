import sys
from collections import defaultdict
from collections import Counter
import math
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Trigram Language Models
Instructor: Daniel Bauer
Student: Yuanyang Teng
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n <= len(sequence).
    """

    ngrams_output = []
    start = 'START'
    stop = 'STOP'

    if n < 1:
        pass

    elif n == 1:
        ngram = (start,)
        ngrams_output.append(ngram)

        for x in sequence:
            ngram = tuple()
            ngram = ngram + (x,)
            ngrams_output.append(ngram)
        ngram = (stop,)
        ngrams_output.append(ngram)

    # 1 < n < len(sequence)
    else:
        # number of ngrams
        for i in range(len(sequence)+1):
            ngram = tuple()
            while len(ngram) < n:
                # construct n-gram
                for j in range(n):
                    if i < (n-1) and j < (n-1-i):
                        ngram = ngram + (start,)
                    elif j < (n-1):
                        ngram = ngram + (sequence[i-(n-j-1)],)
                    elif j == (n-1) and i < len(sequence):
                        ngram = ngram + (sequence[i],)
                    else:
                        ngram = ngram + (stop,)
            ngrams_output.append(ngram)
                        
    return ngrams_output


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {}
        self.bigramcounts = {} 
        self.trigramcounts = {}

        unigrams = []
        bigrams = []
        trigrams = []
        self.wordcount = 0
        self.sentencecount = 0
        for sentence in corpus:
            unigrams += get_ngrams(sentence, 1)
            bigrams += get_ngrams(sentence, 2)
            trigrams += get_ngrams(sentence, 3)
            self.wordcount += len(sentence)
            self.sentencecount += 1

        self.unigramcounts = Counter(unigrams)
        self.bigramcounts = Counter(bigrams)
        self.trigramcounts = Counter(trigrams)

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        if self.bigramcounts[trigram[:2]] == 0:
            count = self.trigramcounts[trigram]
            denominator = self.sentencecount
        else:
            count = self.trigramcounts[trigram]
            denominator = self.bigramcounts[trigram[:2]]
        trigram_raw_prob = count / denominator

        return trigram_raw_prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        if self.unigramcounts[bigram[:1]] == 0:
            count = self.bigramcounts[bigram]
            denominator = self.sentencecount
        else:
            count = self.bigramcounts[bigram]
            denominator = self.unigramcounts[bigram[:1]]
        bigram_raw_prob = count / denominator

        return bigram_raw_prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        count = self.unigramcounts[unigram]
        unigram_raw_prob = count / self.wordcount

        return unigram_raw_prob

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        smooth_prob = lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(trigram[1:]) + lambda3 * self.raw_unigram_probability(trigram[2:])

        return smooth_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        logprob = 0.0
        trigrams_from_sentence = get_ngrams(sentence, 3)
        for x in trigrams_from_sentence:
            smoothed_prob = self.smoothed_trigram_probability(x)
            logprob += math.log2(smoothed_prob)

        return logprob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0.0
        M = 0
        for sentence in corpus:
            l += self.sentence_logprob(sentence)
            M += len(sentence)
        l = (1/M) * l

        perplexity = 2 ** (-l)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total +=1
            if pp < pp2:
                correct += 1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total +=1
            if pp < pp1:
                correct += 1

        accuracy = correct / total

        return accuracy

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # test code
    # seq = ["word", "word2"]
    # n = 3
    # print(get_ngrams(seq, n))

    # model = TrigramModel("brown_train.txt")
    # print(model.unigramcounts)
    # print(model.bigramcounts)
    # print(model.trigramcounts)

    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Training perplexity: ", pp)
    #
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Testing perplexity: ", pp)

    # Essay scoring experiment: 
    acc = essay_scoring_experiment('data/train_high.txt', 'data/train_low.txt', 'data/test_high', 'data/test_low')
    print("Essay Scoring Accuracy: ", acc)

