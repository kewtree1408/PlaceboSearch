#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
from flask import g, Flask, Blueprint, render_template, jsonify, redirect, url_for, current_app, flash, request
from search_engine.utils import build_pager_big
from searcher import finder, get_lst_snippet, DocStat2, get_index
import simplejson
import traceback
import logging
from pymongo import MongoClient

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(process)d.%(name)s.%(levelname)s: %(message)s',
    filename="run_engine.log",
    filemode='a')
log = logging.getLogger()
log.setLevel(logging.DEBUG)


search = Flask(__name__, static_folder='search_engine/static', template_folder='search_engine/templates')
search.dbconnection = MongoClient('localhost', 27017)
search.db = search.dbconnection['placebo']
search.db['queries'].remove()
search.last_queries = search.db['queries']
search.text_sn = search.db['text_for_snippets']
search.rindex = get_index('rindex.pkl', search.text_sn)


@search.route('/')
def index():
    rindex = search.rindex
    query = request.args.get("q", "").strip().lower().strip()
    page = int(request.args.get("p", "1"))
    if page < 1:
        page = 1
    cached = search.last_queries.find_one({"query": query, "page": page})
    n_p = 10
    answers = []
    snippets = []
    res_rank = []
    if query:
        if cached:
            res_rank = pickle.loads(cached['result'].encode('utf8'))
        else:
            res_rank = finder(query, rindex)
            # logging.debug("res_rank => %r", res_rank)
            search.last_queries.insert({"query": query, "result": pickle.dumps(res_rank), "page": page})
        answers = get_lst_snippet(res_rank, search.text_sn, (page-1)*n_p, (page-1)*n_p+n_p)
        # answers = snippets[]

    total = len(res_rank)
    pages = total / n_p + (1 if total % n_p else 0)
    pager = build_pager_big(pages, page, n_p+2)
    return render_template('index.html', page=page, pages=pages, pager=pager,
                           answers=answers, query=query, found=total)


if __name__ == "__main__":
    host, port = '0.0.0.0', 8007
    print 'Server start'
    search.run(host=host, port=port, debug=True)  # add debug