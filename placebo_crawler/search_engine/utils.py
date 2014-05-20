# -*- coding: utf-8 -*-

from flask import render_template, jsonify, redirect, url_for, current_app, flash, request
import pymongo

def append_page_range_custom(left, cnt, pages, current, link_class, current_link_class, link, link_first=None):
    for i in xrange(left, left + cnt):
        if link_first:
            pages.append((link % i if i > 1 else link_first,
                          current_link_class if current == i else link_class,
                          i))
        else:
            pages.append((link % i,
                          current_link_class if current == i else link_class,
                          i))
    return pages

def build_pager_big(pages_count, current_page, maxwidth=11, prewidth=2, postwidth=2):
    if maxwidth < 7:
        maxwidth = 7
    if prewidth < 1:
        prewidth = 1
    if postwidth < 1:
        postwidth = 1
    int_area_min = maxwidth - prewidth - postwidth - 2
    if int_area_min < 3:
        raise ValueError("pager's prewidth (%d) or postwidth (%d) are too big" % (prewidth, postwidth))
    int_area_pre = int_area_min / 2
    int_area_post = int_area_min / 2 if int_area_min % 2 else (int_area_min / 2 - 1)
    print int_area_pre, int_area_post, prewidth, postwidth

    if current_page < 1:
        current_page = 1
    if current_page > pages_count:
        current_page = pages_count
    if pages_count <= maxwidth:
        return append_page_range_custom(1, pages_count, [], current_page, "", "active", "%d")

    skipleft, skipright = False, False
    if current_page <= (int_area_pre + prewidth + 2):
        skipright = True
        left = prewidth + 1
        cnt = int_area_min + 1
    else:
        skipleft = True
        if pages_count - current_page <= (int_area_post + postwidth + 1):
            left = pages_count - postwidth - int_area_min
            cnt = int_area_min + 1
        else:
            skipright = True
            left = current_page - int_area_pre
            cnt = int_area_min

    pages = append_page_range_custom(1, prewidth, [], current_page, "", "active", "%d")
    if skipleft:
        pages.append(("...", '', "..."))
    pages = append_page_range_custom(left, cnt, pages, current_page, "", "active", "%d")
    if skipright:
        pages.append(("...", '', "..."))
    pages = append_page_range_custom(pages_count - postwidth + 1, postwidth, pages, current_page, "", "active", "%d")
    return pages


