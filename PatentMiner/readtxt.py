from unicodecsv import csv
import os


def get_vocab():
    return open_file('vocab')


def get_ignore_word():
    return open_file('ignore_word')


def get_ignore_lemmatize():
    return open_file('ignore_lemmatize')


def open_file(key):
    key_maps = {
        'vocab': 'input/vocab.txt',
        'ignore_word': 'input/ignore_word.txt',
        'ignore_lemmatize': 'input/ignore_lemmatize.txt'
    }
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_name = key_maps.get(key)
    file = os.path.join(ROOT_DIR, file_name)
    with open(file, encoding='utf-8') as source:
        rdr = csv.reader(source)
        words = []
        word_set = set()
        for row in rdr:
            word = row[0].lower()
            if word not in word_set:
                word_set.add(word)
                words.append(word)
    return words
