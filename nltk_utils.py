import numpy as np
import nltk
# nltk.download('punkt')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


from nltk.corpus import wordnet, stopwords
stop_words = set(stopwords.words('english'))
import string

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def lemmatize_word(word):
    return lemmatizer.lemmatize(word.lower())

def remove_stopwords(words):
    return [word for word in words if word not in stop_words and word not in string.punctuation]


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [lemmatize_word(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
