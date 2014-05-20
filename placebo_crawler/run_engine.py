#! /usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, Blueprint, render_template, jsonify, redirect, url_for, current_app, flash, request
from search_engine.utils import build_pager_big
from searcher import finder, get_lst_snippet, DocStat3, DocStat2
import pymongo
import simplejson
import traceback


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
    query = request.args.get("q", "").strip().lower()
    page = int(request.args.get("p", "1"))
    if page < 1:
        page = 1
    results = get_lst_snippet(finder(query))
    total = len(results) #current_app.dbc.prepared_collection.find(where).count()
    n_p = 10
    pages = total / n_p + (1 if total % n_p else 0)
    # cursor = current_app.dbc.prepared_collection.find(where).skip((page - 1) * 15).limit(15)
    # domainlist = []
    # for card in cursor:
    #     addr, addresses = u'', []
    #     phone = ''
    #     if 'addresses' in card and len(card['addresses']):
    #         if len(card['addresses']) == 1:
    #             addr = card['addresses'][0]['address']
    #             phone = card['addresses'][0]['phone']
    #             addresses.append((card['addresses'][0]['address'], card['addresses'][0]['is_head']))
    #         else:
    #             for address in card['addresses']:
    #                 addresses.append((address['address'], address['is_head']))
    #                 if address['is_head']:
    #                     addr = address['address']
    #                     phone = address['phone']
    #             if not addr:
    #                 addr = card['addresses'][0]['address']
    #                 phone = card['addresses'][0]['phone']
    #     domainlist.append((card['domain'], card['title'], card['url'], addr, phone, addresses, is_black))
    try:
        # сюда вставаить кеширование
        answers = []
        if query:
            answers = results[(page-1)*n_p:(page-1)*n_p+n_p]
    except Exception as ex:
        return jsonify(error=traceback.format_exc())
    pager = build_pager_big(pages, page, n_p+2)
    return render_template('index.html', page=page, pages=pages, pager=pager,
                           answers=answers, query=query, found=total)

if __name__ == "__main__":
    search.run(host='0.0.0.0', port='8007')
    # search.debug = True