#! /usr/bin/env python
# coding: utf-8

class TagItems(object):
    tags = {} # { "teg" : [] }

    def __init__(self, stem):

        # Информация
        self.tags['info'] = [stem(u'описание')]

        # Название
        self.tags['name'] = [stem(u'название')]

        # Лекарство
        self.tags['drug'] = [stem(u'лекарство')]

        # Болезнь
        self.tags['disease'] = [stem(u'болезнь')]

        # Состав
        self.tags['description'] = [stem(u'состав'), stem(u'капсулы'), stem(u'раствор'), stem(u'экстракт'), stem(u'флаконы'), stem(u'таблетки'), stem(u'симптомы')]

        # Передозировка
        self.tags['overdose'] = [stem(u'передозировка')]

        # Побочные действия
        self.tags['side'] = [stem(u'побочное'), stem(u'действие')]

        # Способ применения и дозы / показания
        self.tags['usage'] = [stem(u'способ'), stem(u'применения'), stem(u'дозы'), stem(u'показания')]

        # Противопоказания
        self.tags['contra'] = [stem(u'противопоказания')]





