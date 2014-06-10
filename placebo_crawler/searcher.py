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

# from memory_profiler import profile

from pprint import pprint
from math import log, sqrt
import codecs

# rus_stemmer = RussianStemmer()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(u'%(asctime)s - %(message)s')
fh = logging.FileHandler('stats_rindex.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

morph = pymorphy2.MorphAnalyzer()

global_replace = ''

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

    def __init__(self, text_id, labels, freq, posids):
        self.freq = freq
        self.posids = list(posids)
        self.labels = labels
        self.weight = 0
        self.text_id = text_id

    def __str__(self):
        return repr(self)

    def __repr__(self):
        attrs = ('freq', 'posids', 'labels', 'weight', 'text_id')
        args = ", ".join(["%s=%r" % (n, getattr(self, n)) for n in attrs])
        return "%s(%s)" % (self.__class__.__name__, args)

    def __hash__(self):
        return hash(self.text_id)

    @classmethod
    def join(cls, w_ds1, w_ds2):
        w1, ds1 = w_ds1
        w2, ds2 = w_ds2
        posids = ds1.posids
        text_id = ds1.text_id
        if ds1.text_id == ds2.text_id:
            posids = list(set(ds1.posids)|set(ds2.posids))
        elif len(ds2.posids) > (ds1.posids):
            posids = ds2.posids
            text_id = ds2.text_id
        return w1 + w2, cls(
            text_id=text_id,
            labels=sorted(list(set(ds1.labels)|set(ds2.labels))),
            freq=ds1.freq+ds2.freq,
            posids=sorted(posids)
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



def is_punctuation(token, punctuation=set(string.punctuation)):
    for char in token:
        if char in punctuation:
            return True
    return False


# POS - parst of speech - опредление части речи для русского языка
# с помощью библиотеки: http://pymorphy2.readthedocs.org/en/master/user/guide.html
def custom_pos_tag(token):
    p = morph.parse(unicode(token))[0]
    return p.tag.POS


# второстепенные части речи: предлог, междометие, союз, частица, местоимение
def is_minorPOS(token, minors=set([u'PREP', u'INTJ', u'CONJ', u'PRCL', u'NPRO'])):
    if custom_pos_tag(token) in minors:
        return True
    return False


# получаем уникальные токены для текущего документа
def get_tokens(text):
    tokens = collections.defaultdict(lambda: [0, list()])  # {'token': (freq(int), posids(list))}
    pos = 0
    for t in wordpunct_tokenize(text):
        if not is_punctuation(t) and not is_minorPOS(t):
            t_pos = text.find(t, pos)
            pos = t_pos if t_pos != -1 else pos
            tokens[t][0] += 1  # freq + 1
            tokens[t][1].append(pos)  # posids + [pos]
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
def term_ds_from_item(item, db_text, first_tag):
    print "Adding url", item['url']
    for tag in item:
        text = ' ' + item['name'] + ' ' + item[tag]
        terms = get_terms(get_tokens(text))
        text_id = db_text.insert({'text': text, 'url': item['url'], 'title': item['name']})
        for term in terms:
            ds = DocStat2(text_id, [first_tag, tag], *terms[term])
            logger.info("%s %s", term, ds)
            yield term, ds


def build_rindex(db_text):
    rindex = dict()
    for main_tag in ['DRUG', 'DISEASE']:
        fname = "items_%s.pkl" % main_tag
        lower_main_tag = main_tag.lower()
        with open(fname, 'rb') as bf:
            while bf:
                try:
                    obj = cPickle.load(bf)
                    for term, ds in term_ds_from_item(obj, db_text, lower_main_tag):
                        if term not in rindex:
                            rindex[term] = list()
                        rindex[term].append(ds)
                except EOFError as err:
                    print "end load"
                    break

    for t, ds in sorted(rindex.items()):
        logger.debug(u"term = %s, ds = %s", t, str(ds))
    logger.info("Index built, %d items", len(rindex))

    return rindex


# @profile
def get_index(ridx_fname, db_text):
    try:
        with open(ridx_fname, 'rb') as ribf:
            rindex = cPickle.load(ribf)
            print "END load cpickle"
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


def get_term_or_synonym(term, ridx=None, synonyms=None):
    if not synonyms:
        return ridx.get(term, [])
    else:
        result = ridx.get(term, [])
        if result:
            return result
        else:
            for s in synonyms.get(term, []):
                s_norm = morph.parse(unicode(s))[0].normal_form
                result = ridx.get(s_norm, [])

                if result:
                    global global_replace
                    global_replace = s
                    break

            return result


@profile
def get_tf_idf(query, ridx, labels=None, synonyms=None):
    """
    Возвращает список из списков: [
        [tf-idx, [DocStat2_11, DocStat2_21, DocStat2_31, ...]],
        [tf-idx, [DocStat2_12, DocStat2_22, DocStat2_32, ...]],
        ...
    ]
    """
    UP_WEIGHT = 50
    labels = [] if labels is None else list(labels)
    terms_q = dict(get_terms(get_tokens(query)))
    q_docstats = dict()
    N = sum([len(get_term_or_synonym(t, ridx, synonyms)) for t in terms_q])
    print N
    # схема tf-idf для терминов
    term_tf_idf = get_term_tf_idf(terms_q)
    # пересечение документов для всех термов
    intersection_ds = []
    # увеличивает вес
    for t in terms_q:
        # куча документов
        heap_docstats = []
        all_terms_ridx = get_term_or_synonym(t, ridx, synonyms)
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
            if len(heap_docstats) > 500:
                break
        # docstats = sorted([docstat for docstat in all_terms_ridx], key=lambda docst: docst.weight)

        q_docstats[t] = heap_docstats

    if q_docstats:
        intersection_ds = set(reduce(DocStat2.intersection, q_docstats.values()))
    q_sum = sum(term_tf_idf[t] for t in terms_q)

    if intersection_ds:
        for t in terms_q:
            dq_sum = 0.0
            d_sum = 0.0
            qi = term_tf_idf[t]
            for _, ds in intersection_ds:
                if ds.text_id in [docst.text_id for _, docst in q_docstats[t]]:
                    dq_sum += ds.weight*qi
                    d_sum += ds.weight
            if not d_sum:
                continue
            for _, ds in intersection_ds:
                cos_dq = dq_sum/(sqrt(d_sum)*sqrt(q_sum))
                ds.weight = cos_dq
                for label in labels:
                    if label in ds.labels:
                        ds.weight += UP_WEIGHT
                print ds.weight

    rank = collections.defaultdict(lambda: [0, list()])
    for _, ds in intersection_ds:
        rank[ds.text_id][0] += ds.weight
        rank[ds.text_id][1] += [ds]

    for t in q_docstats:
        while q_docstats[t]:
            w, ds = heappop(q_docstats[t])
            if ds.text_id not in [fds.text_id for _, fds in intersection_ds]:
                rank[ds.text_id][0] += w
                rank[ds.text_id][1] += [ds]
            if len(q_docstats) > 500:
                break
    # print rank
    return sorted([rank[d] for d in rank], key=lambda ds: ds[0], reverse=True)


def get_similarity(q, idx, tf_idf):
    COUNT_WORDS_IN_TEXT = 100
    urls_for_sim = set()
    for info in tf_idf:
        weight = info[0]
        ds2_lst = info[1]
        for ds in ds2_lst:
            urls_for_sim.add(ds.text_id)

    rank_lm = []
    est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
    q_sequence = wordpunct_tokenize(q)
    for text_id in urls_for_sim:
        text = ds.text
        sequence = wordpunct_tokenize(text)
        lm = NgramModel(3, sequence, estimator=est)
        ds.weight = lm.entropy(q_sequence)
        # if len(ds.text) > COUNT_WORDS_IN_TEXT:
        rank_lm += [ds]

    return sorted(rank_lm, key=lambda ds: ds.weight)


def finder(q, ridx=None, synonyms=None):
    query, labels = get_tags(q)
    if ridx:
        return get_tf_idf(query, ridx, labels, synonyms)
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
    if pos_end == -1:
        pos_end = text.rfind(' ')
    print pos_begin, pos_end
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
def get_lst_snippet(lst_result, db_text, begin, end):
    snippet = list()
    for res in lst_result[begin:end]:
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
        url = text_for_sn['url']
        title = text_for_sn['title']
        posids = sorted(list(reduce(lambda a, b: set(a) | set(b),
                                    [d.posids for d in lst_ds if d.text_id == ds.text_id])))
        # else:
        #     raise
        # print sn_labels
        labels = [TRANSLATE_LBLs.get(l.strip(), u'') for l in sn_labels]
        # print labels
        short_url = url.split('http://')[1]
        # print short_url
        snippet += [{'url': short_url,
                    'domain': short_url.split('/')[0],
                    'labels': labels,
                    'shorter': snippet_by(text, posids),
                    'title': title}]

    global global_replace
    replace = global_replace
    global_replace = ''
    return snippet, replace


def load_synonyms():
    d = {}
    with codecs.open('drug_synonyms_dictionary.txt', encoding='utf8', mode='r') as file:
        for line in file:
            line = line.split('|')
            name = line[0]
            norm_name = morph.parse(unicode(name))[0].normal_form
            synonyms = line[1].split(',')
            d[norm_name] = [morph.parse(unicode(s))[0].normal_form for s in synonyms[:-1]]
    return d


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

    query = u"печень почки, симптомы; ; : противопоказания,,,"
    synonyms = load_synonyms()
    for s in synonyms:
        print s, synonyms[s]
    # Выясняем, насколько наш запрос соответствует документу
    ridx = get_index('rindex.pkl', db_text)
    res = finder(query, ridx, synonyms)
    print res
    snippets, replace = get_lst_snippet(res, db_text, 0, len(res))
    # for snippet in snippets:
    #     for l in snippet['labels']:
    #         print repr(l)
    #     if len(snippet['shorter']) > 300:
    #         import ipdb; ipdb.set_trace()
    #     print len(snippet['shorter'])


def set_db():
    dbconnection = MongoClient('localhost', 27017)
    db = dbconnection['placebo']
    db['queries'].remove()
    db_text = db['text_for_snippets']
    db_text.ensure_index('id')
    db_text.ensure_index('snippet')
    return db_text


# RINDEX = get_index('rindex.pkl', set_db())


if __name__ == '__main__':
    main()
