# coding: utf-8

import json
import codecs
from scrapy.exceptions import DropItem
from items import DiseaseDescription, DrugDescription, DrugInfo


class JsonWriterPipeline(object):
    """
    Записываем в файл текстовые данные и json
    """
    def __init__(self):
        self.file_txt = codecs.open('items.txt', encoding='utf8', mode='w')
        self.file_json = codecs.open('items.json', encoding='utf8', mode='w')

    def process_item(self, item, spider):
        prefix = ''
        if isinstance(item, DrugInfo) or isinstance(item, DrugDescription):
            prefix = 'DRUG'
        elif isinstance(item, DiseaseDescription):
            prefix = 'DISEASE'

        line = ''.join(['%s_%s: %s\n'%(prefix,k,item[k]) for k in dict(item)])
        json_line = json.dumps(dict(item)) + '\n'
        self.file_txt.write(line)
        self.file_json.write(json_line)
        return item


class DuplicatesPipeline(object):
    """
    Пропускаем дубликаты для описания болезней desiase
    """
    def __init__(self):
        self.ids_seen = set()

    def process_item(self, item, spider):
        if isinstance(item, DiseaseDescription):
            if item['name'] in self.ids_seen:
                raise DropItem("Duplicate item found: %s" % item)
            else:
                self.ids_seen.add(item['name'])
        return item
