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
from tag_items import TagItems

from nltk import wordpunct_tokenize
from nltk.stem.snowball import RussianStemmer
from nltk.probability import LidstoneProbDist
from nltk.model.ngram import NgramModel
from nltk import wordpunct_tokenize
from pymongo import MongoClient
from heapq import heappush, heappop, nlargest
from bson.objectid import ObjectId


from pprint import pprint
from math import log, sqrt

# rus_stemmer = RussianStemmer()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(u'%(asctime)s - %(message)s')
fh = logging.FileHandler('stats_rindex.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

morph = pymorphy2.MorphAnalyzer()


import cProfile
def profile(func):
    """Decorator for run function profile"""
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper


class DocStat2(object):
    def __init__(self, item, text_id, labels, freq, posids, tags=None):
        self.doc_url = item['url'].lstrip('http://')
        self.freq = freq
        self.posids = list(posids)
        self.labels = labels
        self.tags = tags or []
        self.weight = 0
        self.text_id = text_id
        self.title = item['name']

    def __str__(self):
        # posids = ''.join([str(p)+', ' for p in self.posids])
        labels = ''.join([str(l)+', ' for l in self.labels])
        return u'%s: <%d(%f), [%s]>' % (self.doc_url, self.freq, self.weight, labels)

    def __repr__(self):
        return str(self)

    def __eq__(self, ds):
        if self.doc_url == ds.doc_url:
            return True
        return False

    def __hash__(self):
        return hash(self.doc_url)

    @classmethod
    def join(cls, w_ds1, w_ds2):
        w1, ds1 = w_ds1
        w2, ds2 = w_ds2
        posids = ds1.posids
        text_id = ds1.text_id
        if ds1.text_id == ds2.text_id:
            posids = sorted(list(set(ds1.posids)|set(ds2.posids)))
        elif len(ds2.posids) > (ds1.posids):
            posids = ds2.posids
            text_id = ds2.text_id
        return w1 + w2, cls(
            item={'url': ds1.doc_url, 'name': ds1.title},
            text_id=text_id,
            labels=sorted(list(set(ds1.labels)|set(ds2.labels))),
            freq=ds1.freq+ds2.freq,
            posids=posids
        )

    @classmethod
    def intersection(cls, list1, list2):
        list1 = sorted(list(list1), key=lambda x: hash(x))
        list2 = sorted(list(list2), key=lambda x: hash(x))

        if not list1:
            return list2

        if not list2:
            return list1

        result = list()
        current1 = list1.pop()
        current2 = list2.pop()
        while list1 and list2:
            if list1[0] == list2[0]:
                result.append(cls.join(list1.pop(), list2.pop()))
            elif hash(list1[0]) < hash(list2[0]):
                result.append(list1.pop())
            else:
                result.append(list2.pop())

        while list1:
            result.append(list1.pop())

        while list2:
            result.append(list2.pop())

        return result



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
        # term = rus_stemmer.stem(t)
        term = morph.parse(unicode(t))[0].normal_form
        freq, posids = terms[term]
        init_freq, init_posids = tokens[t]
        terms[term] = init_freq + freq, init_posids + posids
    return terms


# @profile
def update_rindex2(rindex, item, db_text, first_tag):
    print "Adding url", item['url']
    tags = item.keys()
    for tag in tags:
        text = item['name'] + ' ' + item[tag]
        terms = get_terms(get_tokens(text))
        text_id = db_text.insert({'text': text, 'url': item['url']})
        for trm in terms:
            ds = DocStat2(item, text_id, [first_tag, tag], *terms[trm])
            rindex[trm].append(ds)



# def update_index3(index, item, key_name='info'):
#     text = item[key_name]
#     ds = DocStat3(item, text)
#     index[ds.doc_url] = ds


# def update_index(rindex2, item, db_text, main_tag):
#     """
#     update rindex:
#     rindex = {
#         token1: [DocStat1, DocStat2, ..., DocStat_N],
#         token2: [DocStat1, DocStat2, ..., DocStat_M],
#         ...
#     }
#     """
    # key_value = ''
    # if main_tag == 'drug':
    #     key_value = 'info'
    # elif main_tag == 'disease':
    #     key_value = 'description'

    # update_rindex2(rindex2, item, db_text, main_tag)


def build_rindex(db_text):
    rindex2 = collections.defaultdict(lambda: list())

    for main_tag in ['DRUG', 'DISEASE']:
        fname = "items_%s.pkl" % main_tag
        lower_main_tag = main_tag.lower()
        with open(fname, 'rb') as bf:
            while bf:
                try:
                    obj = cPickle.load(bf)
                    update_rindex2(rindex2, obj, db_text, lower_main_tag)
                except EOFError as err:
                    print "end load"
                    break
                except Exception as ex:
                    print ex


    for t, ds in sorted(rindex2.items()):
        logger.debug(u"term = %s, ds = %s", t, str(ds))

    return dict(rindex2)


# @profile
def get_index(ridx_fname, db_text):
    try:
        with open(ridx_fname, 'rb') as ribf:
            rindex = cPickle.load(ribf)
    except IOError as ex:
        # удаляем все данные из кеша-монги
        db_text.remove()
        db_text.ensure_index('url')

        # строим индекс
        rindex = build_rindex(db_text)
        with open(ridx_fname, 'wb') as ribf:
            cPickle.dump(rindex, ribf)
    return rindex


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


# @profile
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
    N = sum([len(ridx.get(t, [])) for t in terms_q])
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
            if 'name' in ds.labels:
                ds.weight += 2*UP_WEIGHT
            for label in labels:
                if label in ds.labels:
                    ds.weight += UP_WEIGHT
            ds.weight *= term_tf_idf[t]
            heappush(heap_docstats, (ds.weight, ds))
        # docstats = sorted([docstat for docstat in all_terms_ridx], key=lambda docst: docst.weight)

        q_docstats[t] = heap_docstats

    if q_docstats:
        intersection_ds = reduce(DocStat2.intersection, q_docstats.values())
    q_sum = sum(term_tf_idf[t] for t in terms_q)

    if intersection_ds:
        for t in terms_q:
            dq_sum = 0.0
            d_sum = 0.0
            qi = term_tf_idf[t]
            for _, ds in intersection_ds:
                if ds.doc_url in [docst.doc_url for _, docst in q_docstats[t]]:
                    dq_sum += ds.weight*qi
                    d_sum += ds.weight
            if not d_sum: continue
            for _, ds in intersection_ds:
                cos_dq = dq_sum/(sqrt(d_sum)*sqrt(q_sum))
                ds.weight = cos_dq
                print cos_dq

    rank = collections.defaultdict(lambda: [0, list()])
    for _, ds in intersection_ds:
        rank[ds.doc_url][0] += ds.weight
        rank[ds.doc_url][1] += [ds]
    for t in q_docstats:
        while q_docstats[t]:
            w, ds = heappop(heap_docstats)
            if ds.doc_url not in [fds.doc_url for _, fds in intersection_ds]:
                rank[ds.doc_url][0] += w
                rank[ds.doc_url][1] += [ds]
    # print rank
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
        text = ds.text
        sequence = wordpunct_tokenize(text)
        lm = NgramModel(3, sequence, estimator=est)
        ds.weight = lm.entropy(q_sequence)
        # if len(ds.text) > COUNT_WORDS_IN_TEXT:
        rank_lm += [ds]

    return sorted(rank_lm, key=lambda ds: ds.weight)


def finder(q, ridx=None):
    query, labels = get_tags(q)
    if ridx:
        return get_tf_idf(query, ridx, labels)
    return []

    # пока не понятно, убираем или оставляем похожесть и как ее учитывать?
    # sim = get_similarity(q, idx, tf_idf)
    # print sim

# Ищем теги и удаляем их из запроса, отмечаем что встретился такой тег
def get_tags(q):
    #labels = ['drug', 'overdose']

    labels = []
    stem = lambda word: morph.parse(unicode(word))[0].normal_form  # rus_stemmer.stem(word)
    all_labels = TagItems(stem).tags
    words = wordpunct_tokenize(q)

    clean_q = []

    for word in words:  # разбиваем на слова из запроса
        low_word = word.lower()
        # term = rus_stemmer.stem(low_word)
        term = morph.parse(unicode(low_word))[0].normal_form
        if not is_punctuation(term):
            is_label = False
            for key, value in all_labels.iteritems():  # проходим по словарю тегов
                for synonym in value: # проходим по списку синонимов тега
                    if term == synonym:
                        if not term in labels:
                            labels.append(key)
                            is_label = True
            if not is_label:
                if not word in clean_q:
                    clean_q.append(word)

    str_clean_q = ' '.join(clean_q)
    return str_clean_q, labels


def snippet_by(text, posids):
    SIZE_SNIPPET = 80
    first_pos = posids[0]
    for ws in string.whitespace + ''.join(['_', '*']):
        text = text.replace(ws, ' ')
    pos_begin = text.find(' ', abs(first_pos-SIZE_SNIPPET))
    pos_end = text.find(' ', first_pos+SIZE_SNIPPET)
    print pos_begin, pos_end
    print text
    result_s = ""
    end_word = pos_begin
    for p in posids:
        if pos_begin < p < pos_end:
            result_s += text[end_word:p]
            end_word = text.find(' ', p)
            result_s += text[p:end_word].upper()
    result_s += text[end_word:pos_end]

    return '...'+result_s+'...'


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
    'title': u'название',
    'info': u'описание',
}


# @profile
def get_lst_snippet(lst_result, db_text, begin=0, end=0):
    snippet = list()
    for res in lst_result:
        weight = res[0]
        lst_ds = res[1]
        labels_set = set()
        ds = lst_ds[0]
        for d in lst_ds:
            # if len(d.posids) > len(ds.posids):
                # ds = d
            for l in d.labels:
                labels_set.add(l)

        sn_labels = labels_set if labels_set else ds.labels
        # print sn_labels
        text_for_sn = db_text.find_one({'_id': ObjectId(ds.text_id)})
        # print cur_ds.text_id
        # if text_for_sn.count():
        text = text_for_sn['text']
        posids = sorted(list(reduce(lambda a, b: set(a) | set(b),
                                    [d.posids for d in lst_ds if d.text_id == ds.text_id])))
        # else:
        #     raise
        print sn_labels
        labels = [TRANSLATE_LBLs.get(l.strip(), u'') for l in sn_labels]
        print labels
        snippet += [{'url': ds.doc_url,
                        'domain': ds.doc_url.split('/')[0],
                        'labels': labels,
                        'shorter': snippet_by(text, posids),
                        'title': ds.title}]
    return snippet

# @profile
def main():
    """
    query = u"сердечный спазм,симптомы; ; : противопоказания,,,"
    #labels = ['drug', 'overdose']
    #Выясняем, насколько наш запрос соответствует документу
    finder(query)
    """
    # при обновлении индекса, очищаем кеш запросов
    host, port = '0.0.0.0', 8007

    dbconnection = MongoClient('localhost', 27017)
    db = dbconnection['placebo']
    db['queries'].remove()
    db_text = db['text_for_snippets']
    # db_text.drop_indexes()
    # db_text.remove()
    # db_text.ensure_index('id')
    # db_text.ensure_index('snippet')

    query = u"сердечный спазм,симптомы; ; : противопоказания,,,"
    # Выясняем, насколько наш запрос соответствует документу
    ridx = get_index('rindex.pkl', db_text)
    res = finder(query, ridx)
    snippets = get_lst_snippet(res, db_text)
    for snippet in snippets:
        for l in snippet['labels']:
            print repr(l)
        if len(snippet['shorter']) > 200:
            import ipdb; ipdb.set_trace()
        print len(snippet['shorter'])


class RIndex(object):
    def __init__(self):
        dbconnection = MongoClient('localhost', 27017)
        db = dbconnection['placebo']
        db['queries'].remove()
        self.db_text = db['text_for_snippets']
        self.db_text.ensure_index('id')
        self.db_text.ensure_index('snippet')

    def get_ridx(self):

        return get_index('rindex.pkl', self.db_text)

if __name__ == '__main__':
    main()
