#! /usr/bin/env python
# coding: utf-8

import cPickle
import hashlib
import string
import collections
import logging

from placebo_crawler.items import DrugDescription, DiseaseDescription
from nltk import wordpunct_tokenize
from nltk.stem.snowball import RussianStemmer
from pprint import pprint

# Делаем пока 2 индекса:
# 1. Стандартный, как в лабе. Нужен для подсчета tf-idf
# 2. С метками. Если вдруг в запросе присутсвует одно из ключевых слов-меток, то ищем по 2му индексу
# 3. (если получится). Стром n-грамму. Будем смотреть соответсвие запроса заданному закументу.
# Для каждого документа будет соответсвующая циферка. Этот момент продумать


rus_stemmer = RussianStemmer()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(u'%(asctime)s - %(message)s')
fh = logging.FileHandler('stats_rindex1.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

class DocStat1(object):
    def __init__(self, item, freq, posids):
        self.doc_url = item['url'].lstrip('http://www.')
        self.freq = freq
        self.posids = sorted(posids)

    def __str__(self):
        posids = ''.join([str(p)+', ' for p in self.posids])
        return u'%s: <%d, [%s]>' % (str(self.doc_url), self.freq, posids)

    def __repr__(self):
        return str(self)


class DocStat2(object):
    def __init__(self, item, labels, freq, posids):
        self.doc_url = item['url'].lstrip('http://www.')
        self.freq = freq
        self.posids = posids
        self.labels = labels

    def __str__(self):
        posids = ''.join([str(p)+', ' for p in self.posids])
        labels = ''.join([str(l)+', ' for l in self.labels])
        return u'%s: <%d, [%s], [%s]>' % (self.doc_url, self.freq, posids, labels)

    def __repr__(self):
        return str(self)


def is_punctuation(token):
    for punct in string.punctuation:
        if punct in token:
            return True
    return False


# получаем уникальные токены для текущего документа
def get_tokens(text):
    tokens = collections.defaultdict(lambda: (0, list())) # {'token': (freq(int), posids(list))}
    pos = 0
    for t in wordpunct_tokenize(text):
        if not is_punctuation(t):
            pos += 1
            freq, posids = tokens[t]
            tokens[t] = freq + 1, posids + [pos]
    return tokens


# получаем уникальные термы для текущего документа
def get_terms(tokens):
    terms = collections.defaultdict(lambda: (0, set())) # {'term': (freq(int), posids(list))}
    for t in tokens:
        term = rus_stemmer.stem(t)
        freq, posids = terms[term]
        init_freq, init_posids = tokens[t]
        terms[term] = init_freq + freq, posids | set(init_posids)
    return terms


def update_rindex1(rindex, item):
    """
    @return terms;
    terms = [
        term1: [DocStat1_1, DocStat1_2, ..., DocStat1_N],
        term2: [DocStat1_1, DocStat1_2, ..., DocStat1_M],
        ...
    ]
    """
    text = item['info']
    tokens = get_tokens(text)
    terms = get_terms(tokens)
    for trm in terms:
        rindex[trm] += [DocStat1(item, *terms[trm])]


def update_rindex2(rindex, item, first_tag):
    tags = item.keys()
    tags.remove('info')
    for tag in tags:
        text = item[tag]
        terms = get_terms(get_tokens(text))
        for trm in terms:
            rindex[trm] += [DocStat2(item, [first_tag, tag], *terms[trm])]


def update_indexes(rindex1, rindex2, rindex3, item, main_tag):
    """
    update rindex:
    rindex = {
        token1: [DocStat1, DocStat2, ..., DocStat_N],
        token2: [DocStat1, DocStat2, ..., DocStat_M],
        ...
    }
    """

    text = item['info']
    # update_rindex1(rindex1, item)
    update_rindex2(rindex2, item, main_tag)


def main():
    main_tag = 'DRUG'
    fname = "items_DRUG.pkl"
    rindex1 = collections.defaultdict(lambda: list())
    rindex2 = collections.defaultdict(lambda: list())
    rindex3 = collections.defaultdict(lambda: list())
    with open(fname, 'rb') as bf:
        while bf:
            try:
                obj = cPickle.load(bf)
                print type(obj)
                update_indexes(rindex1, rindex2, rindex3, obj, main_tag)
            except EOFError as err:
                print "end load"
                break
            except Exception as ex:
                print ex
                raise
                continue

    for t, ds in sorted(rindex2.items()):
        print u"term = %s, ds = %s"%(t, str(ds))
        logger.debug(u"term = %s, ds = %s", t, str(ds))

if __name__ == '__main__':
    main()