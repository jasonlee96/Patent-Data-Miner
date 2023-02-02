import nltk
from nltk.tokenize import MWETokenizer, treebank
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import re
from PatentMiner.readtxt import get_ignore_word, get_ignore_lemmatize

# Lemmatizer initialization
lemmatizer = WordNetLemmatizer()

# list of symbols have to remove from the tokens list
symbols = ['*', '(', ')', ':', '-', ',', '.', '!', '?', '_', '/', ';', "'", "[", "]", '{', '}', '+', '&', '%', '#', '@', "<", '>', '|', '*', '\\']

ignore_words = get_ignore_word()
ignore_lemmatize = get_ignore_lemmatize()

# stopwords list initialization
stopwords_list = stopwords.words("english")
stopwords_list.extend(symbols)
stopwords_list.extend(ignore_words)

# Multi word expression that used to merge the separated token
mwe_list = []
# Retokenize by merge those keywords from different tokens such as ('c', '#') -> ('c#')
retokenizer = MWETokenizer(mwes=mwe_list, separator='')

url_pattern = re.compile(r'(?:http(s)?:\/\/)?([\w.-]+(?:\.[\w.-]+))+[\w\-.:/#?[\]@!$&()*+~_]+')

# function check_unicode
# params: word : str
# description: remove those unicode that can't encode into ascii code, such as chinese word and some special character
# return bool
def check_unicode(word) -> bool:
    try:
        word.encode('ascii').decode('utf-8')
        return True
    except UnicodeEncodeError:
        return False
    except UnicodeDecodeError:
        return False


# function get_pos
# params: word : str
# description: to get the pos tag of a word. Such as adj, verb, noun, adverb
# return wordnet object
def get_pos(word) -> wordnet:
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_maps = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_maps.get(tag, wordnet.NOUN)  # get pos tag with default value of Noun.


# function process_multiple
# params: word_list : list[str]
# description: remove stopword and lemmatize the word into its original form from the list
# return list of processed word
def process_multiple(word_list) -> list:
    lemmatized_list = []
    for word in word_list:
        if len(word) == 0:
            continue
        if word in ignore_lemmatize:
            lemmatized_list.append(word)
        else:
            lemmatized_list.append(lemmatizer.lemmatize(word, pos=get_pos(word)))

    # Remove stopwords from the list
    filtered_list = [word for word in lemmatized_list if word not in stopwords_list]
    # Remove empty string in the token list
    filtered_list = filter(None, filtered_list)
    return list(filtered_list)


# function process_single
# params: word : str
# description: remove stopword and lemmatize the word into its original form from a string
# return string or None
def process_single(word) -> str:
    if word not in ignore_lemmatize:
        word = lemmatizer.lemmatize(word, pos=get_pos(word))
    if word not in stopwords_list:
        return word


def tokenize(text):
    # Eliminate url string
    text = re.sub(url_pattern, '', text)
    tokens = retokenizer.tokenize(nltk.word_tokenize(text.lower()))
    prepared_token = []
    for word in tokens:
        if not check_unicode(word):
            continue
        if "\\" in word or "*" in word or "'" in word or "=" in word or "|" in word:
            continue
        if word.startswith('-'):
            word = word.replace('-', '')
        if word.startswith('+'):
            word = word.replace('+', '')
        if re.match(r'(\W+\d+)|(\d)', word):
            continue
        # Try to split the joined tokens such as ("Design/Test") into two different token
        if "/" in word:
            if len(word) == 1:
                continue
            prepared_token.extend(process_multiple(word.split("/")))
            continue
        if len(word) == 0:
            continue
        if "-" in word:
            prepared_token.extend(process_multiple(word.split("-")))
            continue
        
        process_word = process_single(word)
        if process_word is not None:
            prepared_token.append(process_word)

    return prepared_token
