# ngram model
This project builds a trigram language model, which is then applied to a text classification task to evaluate ETS TOEFL essays. The probability distributions are not pre-computed, instead, the model stores the raw counts of ngram occurrences and computes probabilities on demand. The model is evaluated using perplexity on an entire corpus. The project is composed of the following parts: 
- extract ngrams from a sentence
- counting ngram in a corpus
- raw ngram probabilities
- smoothed probabilities
- computing sentence probability
- perplexity



## Usage 
```python
python ngram_model.py data/brown_train.txt data/brown_test.txt
```
## Output
```python
Training perplexity:  18.02694455227229
Testing perplexity:  300.17653468276933
Essay Scoring Accuracy:  0.848605577689243
```
## Data
The ETS data set is proprietary and licensed to Columbia University for research and educational use only.

