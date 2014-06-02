#! /usr/bin/env python
# coding: utf-8

import cPickle
import hashlib
import string
import collections
import logging
import simplejson

from placebo_crawler.items import DrugDescription, DiseaseDescription

from nltk import wordpunct_tokenize
from nltk.stem.snowball import RussianStemmer
from nltk.probability import LidstoneProbDist
from nltk.model.ngram import NgramModel
from pymongo import MongoClient


from pprint import pprint
from math import log


rus_stemmer = RussianStemmer()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(u'%(asctime)s - %(message)s')
fh = logging.FileHandler('stats_rindex.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)



class DocStat1(object):
    def __init__(self, item, freq, posids):
        self.doc_url = item['url'].lstrip('http://www.')
        self.freq = freq
        self.posids = list(posids)
        self.weight = 0

    def __str__(self):
        posids = ''.join([str(p)+', ' for p in self.posids])
        return u'%s: <%d(%f), [%s]>' % (str(self.doc_url), self.freq, self.weight, posids)

    def __repr__(self):
        return str(self)


class DocStat2(object):
    def __init__(self, item, labels, text, freq, posids):
        self.doc_url = item['url'].lstrip('http://www.')
        self.freq = freq
        self.posids = list(posids)
        self.labels = labels
        self.weight = 0
        self.text = text
        self.title = item['name']

    def __str__(self):
        posids = ''.join([str(p)+', ' for p in self.posids])
        labels = ''.join([str(l)+', ' for l in self.labels])
        return u'%s: <%d(%f), [%s], [%s]> <text...%d>' % (self.doc_url, self.freq, self.weight, posids, labels, len(self.text))

    def __repr__(self):
        return str(self)


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
    tokens = collections.defaultdict(lambda: (0, list()))  # {'token': (freq(int), posids(list))}
    pos = 0
    for t in wordpunct_tokenize(text):
        if not is_punctuation(t):
            t_pos = text.find(t, pos)
            pos = t_pos if t_pos != -1 else pos
            freq, posids = tokens[t]
            tokens[t] = freq + 1, posids + [pos]
    return tokens


# получаем уникальные термы для текущего документа
def get_terms(tokens):
    terms = collections.defaultdict(lambda: (0, list())) # {'term': (freq(int), posids(list))}
    for t in tokens:
        term = rus_stemmer.stem(t)
        freq, posids = terms[term]
        init_freq, init_posids = tokens[t]
        terms[term] = init_freq + freq, sorted(init_posids + posids)
    return terms


def update_rindex2(rindex, item, first_tag, key_name='info'):
    tags = item.keys()
    tags.remove(key_name)
    for tag in tags:
        text = item[tag]
        terms = get_terms(get_tokens(text))
        for trm in terms:
            # print text
            rindex[trm] += [DocStat2(item, [first_tag, tag], text, *terms[trm])]


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
    key_value = ''
    if main_tag == 'drug':
        key_value = 'info'
    elif main_tag == 'disease':
        key_value = 'description'

    update_rindex2(rindex2, item, main_tag, key_value)
    update_index3(index3, item, key_value)


def build_rindex():
    rindex1 = collections.defaultdict(lambda: list())
    rindex2 = collections.defaultdict(lambda: list())
    index3 = dict()

    for main_tag in ['DRUG', 'DISEASE']:
        fname = "items_%s.pkl" % main_tag
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

    for t, ds in sorted(rindex2.items()):
        # print u"term = %s, ds = %s"%(t, str(ds))
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


def get_tf_idf(q, ridx, labels=None):
    """
    Возвращает список из списков: [
        [tf-idx, [DocStat2_11, DocStat2_21, DocStat2_31, ...]],
        [tf-idx, [DocStat2_12, DocStat2_22, DocStat2_32, ...]],
        ...
    ]
    """
    BIG_WEIGHT = 10.0
    labels = [] if labels is None else list(labels)
    terms_q = dict(get_terms(get_tokens(q)))
    q_docstats = []
    N = sum([len(ridx.get(t,[])) for t in terms_q])
    print N
    # TODO: нужно сделать схему tf-idf для термов в запросе
    # TODO: Сделать нормализацию по векторам. Считать косунус угла между нормированными векторами
    for t in terms_q:
        df = len(ridx.get(t, []))
        idf = log(1.0*N/df) if df != 0 else 0
        for ds in ridx.get(t, []):
            ds.weight = (1.0+log(ds.freq))*idf
            # учитываем метки
            for label in labels:
                if label in ds.labels:
                    ds.weight += BIG_WEIGHT

        q_docstats += sorted([docstat for docstat in ridx.get(t, [])], key=lambda ds: ds.weight)

    rank = collections.defaultdict(lambda: [0, list()])
    for ds in q_docstats:
        rank[ds.doc_url][0] += ds.weight
        rank[ds.doc_url][1] += [ds]

    return sorted([rank[d] for d in rank], key=lambda ds: ds[0], reverse=True)


def get_similarity(q, idx, tf_idf):
    COUNT_WORDS_IN_TEXT = 100
    urls_for_sim = set()
    for info in tf_idf:
        weight = info[0]
        ds2_lst = info[1]
        for ds in ds2_lst:
            urls_for_sim.add(ds.doc_url)

    rank_lm = []
    est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
    q_sequence = wordpunct_tokenize(q)
    for doc_url in urls_for_sim:
        ds = idx[doc_url]
        text = ds.text
        sequence = wordpunct_tokenize(text)
        lm = NgramModel(3, sequence, estimator=est)
        ds.weight = lm.entropy(q_sequence)
        # if len(ds.text) > COUNT_WORDS_IN_TEXT:
        rank_lm += [ds]

    return sorted(rank_lm, key=lambda ds: ds.weight)


def finder(q, labels=None):
    ridx, inx = get_indexes('rindex.pkl', 'index.pkl')
    tf_idf = get_tf_idf(q, ridx, labels)
    # пока не понятно, убираем или оставляем похожесть и как ее учитывать?
    # sim = get_similarity(q, idx, tf_idf)
    # print sim
    return tf_idf


def snippet_by(ds):
    SIZE_SNIPPET = 80
    if not isinstance(ds.posids, list):
        ds.posids = list(ds.posids)
    pos1 = ds.posids[0]
    text = ds.text
    pos = 0
    begin_pos = pos1
    second_pos = 0

    for t in wordpunct_tokenize(text):
        if text[pos].isupper():
            begin_pos = pos
        if is_punctuation(text[pos]):
            second_pos = pos
        if pos - begin_pos > SIZE_SNIPPET/2:
            begin_pos = second_pos
        if pos < pos1 and pos - begin_pos < SIZE_SNIPPET/2:
            # space_pos = text.rfind(' ', 0, begin_pos-1)
            text_without_endspaces = text[begin_pos:pos1] + '$' + text[pos1:pos1+SIZE_SNIPPET].strip()
            text_without_lf = text_without_endspaces.replace('\n', '; ')
            snippet = text_without_lf + "..."
            return snippet
        pos += 1


def get_snippet(lst_result):
    for res in lst_result:
        weight = res[0]
        lst_ds = res[1]
        ds = lst_ds[0]
        yield snippet_by(ds)


TRANSLATE_LBLs = {
    'description': u'описание',
    'drugs': u'лекарство',
    'drug': u'лекарство',
    'classification': u'классификация',
    'usage': u'показания',
    'contra': u'противопоказания',
    'side': u'побочные действия',
    'overdose': u'передозировка',
    'name': u'название',
}

def get_lst_snippet(lst_result):
    snippet = []
    for res in lst_result:
        weight = res[0]
        lst_ds = res[1]
        ds = lst_ds[0]
        labels = [TRANSLATE_LBLs.get(l.strip(), u'') for l in ds.labels]
        snippet.append({'url': ds.doc_url,
                        'domain': ds.doc_url.split('/')[0],
                        'labels': labels,
                        'shorter': snippet_by(ds),
                        'title': ds.title})
    return snippet


def main():
    # query = u"сердечный спазм"
    # labels = ['drug', 'overdose']
    # Выясняем, насколько наш запрос соответствует документу
    # finder(query)

    res = get_indexes('rindex.pkl', 'index.pkl')
    print res
    # при обновлении индекса, очищаем кеш
    dbconnection = MongoClient('localhost', 27017)
    db = dbconnection['placebo']
    db['queries'].remove()



if __name__ == '__main__':
    main()