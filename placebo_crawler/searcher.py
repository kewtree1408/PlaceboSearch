#! /usr/bin/env python
# coding: utf-8

import cPickle
import hashlib
import string
import collections
import logging
import simplejson
import os
import subprocess

import pymorphy2
from os.path import join

from placebo_crawler.items import DrugDescription, DiseaseDescription

from nltk import wordpunct_tokenize
from nltk.stem.snowball import RussianStemmer
from nltk.probability import LidstoneProbDist
from nltk.model.ngram import NgramModel
from pymongo import MongoClient
from heapq import heappush, heappop, nlargest


from pprint import pprint
from math import log, sqrt

rus_stemmer = RussianStemmer()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(u'%(asctime)s - %(message)s')
fh = logging.FileHandler('stats_rindex.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# в morph_dicts хранятся все словари
# def download_morph():
#     # скачиваем и используем словарь для получения грамматической информации о слове (часть речи)
#     path_to_dictionary = os.path.realpath(os.path.curdir)
#     morph_path = join(path_to_dictionary, 'morph_dicts')
#     if not os.path.exists(morph_path):
#         subprocess.call(['wget', 'https://bitbucket.org/kmike/pymorphy/downloads/ru.sqlite-json.zip'])
#         subprocess.call(['unzip', 'ru.sqlite-json.zip', '-d', 'morph_dicts'])
#     morph = get_morph(morph_path)
#     return morph

morph = pymorphy2.MorphAnalyzer()


class DocStat2(object):
    def __init__(self, item, labels, text, freq, posids):
        self.doc_url = item['url'].lstrip('http://')
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

    def __eq__(self, ds):
        if self.doc_url == ds.doc_url:
            return True
        return False

    def __hash__(self):
        return hash(self.doc_url)


# class DocStat3(object):
#     def __init__(self, item, text):
#         self.doc_url = item['url'].lstrip('http://www.')
#         self.text = text
#         self.weight = 0.0
#
#     def __str__(self):
#         return u'%s: <%d ..., (%f)>' % (self.doc_url, len(self.text), self.weight)
#
#     def __repr__(self):
#         return str(self)


def is_punctuation(token):
    for punct in string.punctuation:
        if punct in token:
            return True
    return False


# POS - parst of speech - опредление части речи для русского языка
# с помощью библиотеки: http://pymorphy2.readthedocs.org/en/master/user/guide.html
def custom_pos_tag(token):
    global morph
    p = morph.parse(unicode(token))[0]
    return p.tag.POS


# второстепенные части речи: предлог, междометие, союз, частица, местоимение
def is_minorPOS(token):
    if custom_pos_tag(token) in [u'PREP', u'INTJ', u'CONJ', u'PRCL', u'NPRO']:
        return True
    return False


# получаем уникальные токены для текущего документа
def get_tokens(text):
    tokens = collections.defaultdict(lambda: (0, list()))  # {'token': (freq(int), posids(list))}
    pos = 0
    for t in wordpunct_tokenize(text):
        if not is_punctuation(t) and not is_minorPOS(t):
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


def update_rindex2(rindex, item, first_tag):
    tags = item.keys()
    for tag in tags:
        text = item[tag]
        terms = get_terms(get_tokens(text))
        for trm in terms:
            rindex[trm] += [DocStat2(item, [first_tag, tag], text, *terms[trm])]


# def update_index3(index, item, key_name='info'):
#     text = item[key_name]
#     ds = DocStat3(item, text)
#     index[ds.doc_url] = ds


def update_index(rindex2, item, main_tag):
    """
    update rindex:
    rindex = {
        token1: [DocStat1, DocStat2, ..., DocStat_N],
        token2: [DocStat1, DocStat2, ..., DocStat_M],
        ...
    }
    """
    # key_value = ''
    # if main_tag == 'drug':
    #     key_value = 'info'
    # elif main_tag == 'disease':
    #     key_value = 'description'

    update_rindex2(rindex2, item, main_tag)


def build_rindex():
    rindex2 = collections.defaultdict(lambda: list())

    for main_tag in ['DRUG', 'DISEASE']:
        fname = "items_%s.pkl" % main_tag
        with open(fname, 'rb') as bf:
            while bf:
                try:
                    obj = cPickle.load(bf)
                    update_index(rindex2, obj, main_tag.lower())
                except EOFError as err:
                    print "end load"
                    break
                except Exception as ex:
                    print ex
                    raise

    for t, ds in sorted(rindex2.items()):
        logger.debug(u"term = %s, ds = %s", t, str(ds))

    return dict(rindex2)

def get_index(ridx_fname):
    try:
        with open(ridx_fname, 'rb') as ribf:
            rindex = cPickle.load(ribf)
    except IOError as ex:
        rindex = build_rindex()
        with open(ridx_fname, 'wb') as ribf:
            cPickle.dump(rindex, ribf)
    return rindex


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


def get_term_tf_idf(terms_q):
    """
    tf-idf для терминов
    """
    t_w = dict()
    N = len(terms_q)
    for t in terms_q:
        df = terms_q[t][0]
        idf = log(1.0*N/df)
        w = (1.0+log(df))*idf
        t_w[t] = w
    return t_w


def get_tf_idf(query, ridx, labels=None):
    """
    Возвращает список из списков: [
        [tf-idx, [DocStat2_11, DocStat2_21, DocStat2_31, ...]],
        [tf-idx, [DocStat2_12, DocStat2_22, DocStat2_32, ...]],
        ...
    ]
    """
    UP_WEIGHT = 2
    labels = [] if labels is None else list(labels)
    terms_q = dict(get_terms(get_tokens(query)))
    q_docstats = dict()
    N = sum([len(ridx.get(t,[])) for t in terms_q])
    print N
    # схема tf-idf для терминов
    term_tf_idf = get_term_tf_idf(terms_q)
    # пересечение документов для всех термов
    intersection_ds = []
    # куча документов
    heap_docstats = []
    # увеличивает вес
    for t in terms_q:
        all_terms_ridx = ridx.get(t, [])
        df = len(all_terms_ridx)
        idf = log(1.0*N/df) if df != 0 else 0
        for ds in all_terms_ridx:
            ds.weight = (1.0+log(ds.freq))*idf
            #  учитываем метки
            for label in labels:
                if label in ds.labels:
                    ds.weight += UP_WEIGHT
            ds.weight *= term_tf_idf[t]
            # heappush(heap_docstats, (ds.weight, ds))
        docstats = sorted([docstat for docstat in all_terms_ridx], key=lambda docst: docst.weight)

        q_docstats[t] = docstats
    if q_docstats:
        intersection_ds = reduce(set.intersection, [set(q_docstats[t]) for t in q_docstats])
    q_sum = sum(term_tf_idf[t] for t in terms_q)

    if intersection_ds:
        for t in terms_q:
            dq_sum = 0.0
            d_sum = 0.0
            qi = term_tf_idf[t]
            for ds in intersection_ds:
                if ds.doc_url in [docst.doc_url for docst in q_docstats[t]]:
                    dq_sum += ds.weight*qi
                    d_sum += ds.weight
            if not d_sum: continue
            for ds in intersection_ds:
                cos_dq = dq_sum/(sqrt(d_sum)*sqrt(q_sum))
                ds.weight = cos_dq
                print cos_dq

    rank = collections.defaultdict(lambda: [0, list()])
    for ds in intersection_ds:
        rank[ds.doc_url][0] += ds.weight
        rank[ds.doc_url][1] += [ds]
    for t in q_docstats:
        for ds in q_docstats[t]:
            if ds.doc_url not in [fds.doc_url for fds in intersection_ds]:
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


def finder(q, labels=None, ridx=None):
    if ridx:
        return get_tf_idf(q, ridx, labels)
    return []

    # пока не понятно, убираем или оставляем похожесть и как ее учитывать?
    # sim = get_similarity(q, idx, tf_idf)
    # print sim


def snippet_by(ds):
    SIZE_SNIPPET = 80
    if not isinstance(ds.posids, list):
        ds.posids = list(ds.posids)
    pos1 = ds.posids[0]
    text = ds.text
    pos = 0
    begin_pos = pos1
    second_pos = 0
    BORDER = 100

    pos2 = text.find(' ', pos1)
    return text[pos1-BORDER:pos1] + text[pos1:pos2].upper() + text[pos2:pos2+BORDER]

    # for t in wordpunct_tokenize(text):
    #     if text[pos].isupper():
    #         begin_pos = pos
    #     if is_punctuation(text[pos]):
    #         second_pos = pos
    #     if pos - begin_pos > SIZE_SNIPPET/2:
    #         begin_pos = second_pos
    #     if pos < pos1 and pos - begin_pos < SIZE_SNIPPET/2:
    #         # space_pos = text.rfind(' ', 0, begin_pos-1)
    #         text_without_endspaces = text[begin_pos:pos1] + '$' + text[pos1:pos1+SIZE_SNIPPET].strip()
    #         text_without_lf = text_without_endspaces.replace('\n', '; ')
    #         snippet = text_without_lf + "..."
    #         return snippet
    #     pos += 1


def get_snippet(lst_result, out_labels):
    for res in lst_result:
        weight = res[0]
        lst_ds = res[1]
        labels_set = set()
        cur_ds = None
        for ds in lst_ds:
            for l in ds.labels:
                if l in out_labels:
                    labels_set.add(l)
                cur_ds = ds
        cur_ds.labels = labels_set
        yield snippet_by(cur_ds)


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
    'disease': u'заболевание',
}


def get_lst_snippet(lst_result, out_labels):
    snippet = []
    for res in lst_result:
        weight = res[0]
        lst_ds = res[1]
        labels_set = set()
        cur_ds = None
        for ds in lst_ds:
            for l in ds.labels:
                if l in out_labels:
                    labels_set.add(l)
                cur_ds = ds

        sn_labels = labels_set if labels_set else lst_ds[0].labels
        print sn_labels
        labels = [TRANSLATE_LBLs.get(l.strip(), u'') for l in sn_labels]
        snippet.append({'url': cur_ds.doc_url,
                        'domain': cur_ds.doc_url.split('/')[0],
                        'labels': labels,
                        'shorter': snippet_by(cur_ds),
                        'title': cur_ds.title})
    return snippet


def main():
    query = u"сердечный приступ"
    labels = ['contra', 'overdose']
    # Выясняем, насколько наш запрос соответствует документу
    ridx = get_index('rindex.pkl')
    res = finder(query, labels, ridx)
    snippets = get_lst_snippet(res, labels)
    for snippet in snippets:
        for l in snippet['labels']:
            print l,
        print snippet['shorter']

    # при обновлении индекса, очищаем кеш
    dbconnection = MongoClient('localhost', 27017)
    db = dbconnection['placebo']
    db['queries'].remove()


if __name__ == '__main__':
    main()
