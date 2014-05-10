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
from nltk.probability import LidstoneProbDist
from nltk.model.ngram import NgramModel

from pprint import pprint
from math import log

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
        self.weight = 0

    def __str__(self):
        posids = ''.join([str(p)+', ' for p in self.posids])
        return u'%s: <%d(%f), [%s]>' % (str(self.doc_url), self.freq, self.weight, posids)

    def __repr__(self):
        return str(self)


class DocStat2(object):
    def __init__(self, item, labels, freq, posids):
        self.doc_url = item['url'].lstrip('http://www.')
        self.freq = freq
        self.posids = posids
        self.labels = labels
        self.weight = 0

    def __str__(self):
        posids = ''.join([str(p)+', ' for p in self.posids])
        labels = ''.join([str(l)+', ' for l in self.labels])
        return u'%s: <%d(%f), [%s], [%s]>' % (self.doc_url, self.freq, self.weight, posids, labels)

    def __repr__(self):
        return str(self)

    # def __qt__(self, ds):
    #     if self.weight > ds.weight:
    #         return self
    #     return ds
    #
    # def __lt__(self, ds):
    #     return not self.__qt__(ds)


class DocStat3(object):
    def __init__(self, item, text):
        self.doc_url = item['url'].lstrip('http://www.')
        self.text = text
        self.weight = 0.0

    def __str__(self):
        return u'%s: <%d ..., (%f)>' % (self.doc_url, len(self.text), self.weight)

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


def update_rindex1(rindex, item, key_name='info'):
    """
    @return terms;
    terms = [
        term1: [DocStat1_1, DocStat1_2, ..., DocStat1_N],
        term2: [DocStat1_1, DocStat1_2, ..., DocStat1_M],
        ...
    ]
    """
    text = item[key_name]
    tokens = get_tokens(text)
    terms = get_terms(tokens)
    for trm in terms:
        rindex[trm] += [DocStat1(item, *terms[trm])]


def update_rindex2(rindex, item, first_tag, key_name='info'):
    tags = item.keys()
    tags.remove(key_name)
    for tag in tags:
        text = item[tag]
        terms = get_terms(get_tokens(text))
        for trm in terms:
            rindex[trm] += [DocStat2(item, [first_tag, tag], *terms[trm])]


def update_index3(index, item, key_name='info'):
    text = item[key_name]
    ds = DocStat3(item, text)
    index[ds.doc_url] = ds


def update_indexes(rindex1, rindex2, index3, item, main_tag):
    """
    update rindex:
    rindex = {
        token1: [DocStat1, DocStat2, ..., DocStat_N],
        token2: [DocStat1, DocStat2, ..., DocStat_M],
        ...
    }
    """
    if main_tag == 'drug':
        key_value = 'info'
    elif main_tag == 'disease':
        key_value = 'description'
    # update_rindex1(rindex1, item)

    update_rindex2(rindex2, item, main_tag, key_value)
    update_index3(index3, item, key_value)


def build_rindex():
    main_tag = 'DRUG'
    fname = "items_DRUG.pkl"
    rindex1 = collections.defaultdict(lambda: list())
    rindex2 = collections.defaultdict(lambda: list())
    index3 = dict()

    for main_tag in ['DRUG', 'DISEASE']:
        fname = "items_%s.pkl"%main_tag
        with open(fname, 'rb') as bf:
            while bf:
                try:
                    obj = cPickle.load(bf)
                    update_indexes(rindex1, rindex2, index3, obj, main_tag.lower())
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

    return dict(rindex2), dict(index3)

def get_indexes(ridx_fname, idx_fname):
    try:
        with open(idx_fname, 'rb') as ibf:
            index = cPickle.load(ibf)
        with open(ridx_fname, 'rb') as ribf:
            rindex = cPickle.load(ribf)
    except IOError as ex:
        rindex, index = build_rindex()
        with open(idx_fname, 'wb')  as ibf:
            cPickle.dump(index, ibf)
        with open(ridx_fname, 'wb') as ribf:
            cPickle.dump(rindex, ribf)
    return rindex, index


# def finder(q, ridx):
#     terms_q = dict(get_terms(get_tokens(q)))
#     rank = dict()
#     for t in terms_q:
#         rank[t] = sorted([docstat for docstat in ridx.get(t, [])], key=lambda ds: 1 + log(ds.freq))
#         print t
#         print rank[t]
#     # пересечение координатных блоков через intersection
#     res = reduce(set.intersection, [set(rank[r]) for r in rank])
#     print res


def get_tf_idf(q, ridx):
    """
    Возвращает список из списков: [
        [tf-idx, [DocStat2_11, DocStat2_21, DocStat2_31, ...]],
        [tf-idx, [DocStat2_12, DocStat2_22, DocStat2_32, ...]],
        ...
    ]
    """
    terms_q = dict(get_terms(get_tokens(q)))
    q_docstats = []
    N = sum([len(ridx.get(t,[])) for t in terms_q])
    print N
    for t in terms_q:
        df = len(ridx.get(t, []))
        idf = log(1.0*N/df) if df != 0 else 0
        for ds in ridx.get(t, []):
            ds.weight = (1.0+log(ds.freq))*idf
        q_docstats += sorted([docstat for docstat in ridx.get(t, [])], key=lambda ds: ds.weight)

    rank = collections.defaultdict(lambda: [0, list()])
    for ds in q_docstats:
        rank[ds.doc_url][0] += ds.weight
        rank[ds.doc_url][1] += [ds]

    return sorted([rank[d] for d in rank], key=lambda ds: ds[0], reverse=True)


def get_similarity(q, idx):
    rank_lm = []
    est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
    q_sequence = wordpunct_tokenize(q)
    for doc_url in idx:
        ds = idx[doc_url]
        text = ds.text
        sequence = wordpunct_tokenize(text)
        lm = NgramModel(3, sequence, estimator=est)
        ds.weight = lm.entropy(q_sequence)
        rank_lm += [ds]

    return sorted(rank_lm, key=lambda ds: ds.weight)


def finder(q, ridx, idx):
    get_tf_idf(q, ridx)
    sim = get_similarity(q, idx)
    return sim


def main():
    rindex, index = get_indexes('rindex.pkl', 'index.pkl')
    query = u"острая боль в горле"

    # Выясняем, насколько наш запрос соответствует
    res = finder(query, rindex, index)

    print res





if __name__ == '__main__':
    main()