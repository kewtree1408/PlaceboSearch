# coding: utf-8

import json
import codecs
import cPickle

from scrapy.exceptions import DropItem
from items import DiseaseDescription, DrugDescription


class JsonWriterPipeline(object):
    """
    Записываем в файл текстовые данные и pkl
    """
    def __init__(self):
        self.drug_file_txt = codecs.open('items_DRUG.txt', encoding='utf8', mode='aw')
        self.drug_file_txt.write("====================================================\n")
        self.disease_file_txt = codecs.open('items_DISEASE.txt', encoding='utf8', mode='aw')
        self.disease_file_txt.write("====================================================\n")

        self.drug_pkl = open('items_DRUG.pkl', 'aw')
        self.disease_pkl = open('items_DRUG.pkl', 'aw')


    def process_item(self, item, spider):
        line = ''.join(['%s: %s\n'%(k,item[k]) for k in dict(item)])
        
        if isinstance(item, DrugDescription):
            fl_txt = self.drug_file_txt
            fl_pkl = self.drug_pkl
        elif isinstance(item, DiseaseDescription):
            fl_txt = self.disease_file_txt
            fl_pkl = self.disease_pkl
        
        fl_txt.write(line)
        cPickle.dump(dict(item),fl_pkl)
        
        return item


class DuplicatesPipeline(object):
    """
    Не допускает 2х одинаковых документов (url'ов)
    """
    def __init__(self):
        self.ids_seen = set()

    def process_item(self, item, spider):
        if item['url'] in self.ids_seen:
            raise DropItem("Duplicate item found: %s" % item)
        else:
            self.ids_seen.add(item['url'])
            return item


