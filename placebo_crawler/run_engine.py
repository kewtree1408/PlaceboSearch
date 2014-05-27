#! /usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, Blueprint, render_template, jsonify, redirect, url_for, current_app, flash, request
from search_engine.utils import build_pager_big
from searcher import finder, get_lst_snippet, DocStat3, DocStat2
import simplejson
import traceback
from pymongo import MongoClient


search = Flask(__name__, static_folder='search_engine/static', template_folder='search_engine/templates')

@search.route('/api/<query>', methods=['GET','POST'])
def finder_q(query):
    try:
        results = get_lst_snippet(finder(query))
    except Exception as ex:
        return jsonify(result="error", error=traceback.format_exc())
    return simplejson.dumps(results, indent=2, encoding='utf-8')


@search.route('/')
def index():
    try:
        query = request.args.get("q", "").strip().lower().strip()
        page = int(request.args.get("p", "1"))
        if page < 1:
            page = 1

        drugs = request.args.get("drugs", "")
        # disease = request.args.get("disease", "")
        cursor = search.last_queries.find({query: {"$exists": "true"}})
        labels = [drugs]
        n_p = 10
        answers = []
        snippets = []
        if query:
            if cursor.count() > 0:
                snippets = cursor[0][query]
            else:
                snippets = get_lst_snippet(finder(query, labels))
                search.last_queries.insert({query: snippets})
            answers = snippets[(page-1)*n_p:(page-1)*n_p+n_p]

        total = len(snippets)
        pages = total / n_p + (1 if total % n_p else 0)
        pager = build_pager_big(pages, page, n_p+2)
        return render_template('index.html', page=page, pages=pages, pager=pager,
                               answers=answers, query=query, found=total,
                               drugs=drugs)
    except Exception as ex:
        return jsonify(error=traceback.format_exc())


if __name__ == "__main__":
    host, port = '0.0.0.0', '8007'
    search.dbconnection = MongoClient('localhost', 27017)
    search.db = search.dbconnection['placebo']
    search.last_queries = search.db['queries']
    search.run(host=host, port=port) #add debug