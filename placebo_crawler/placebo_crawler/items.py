# coding: utf-8

from scrapy.item import Item, Field

class DrugInfo(Item):
    """
    Вся информация на странице (без категорий)
    name - название лекарства
    info - вся информация
    """
    url = Field()
    name = Field()
    info = Field()


class DrugDescription(Item):
    """
    Полное описание лекарства:
    - классификация по болезням
    - применение
    - противопоказания
    - побочные действия
    - передозировка
    """
    url = Field()
    name = Field()
    classification = Field()
    description = Field()
    usage = Field()
    contra = Field()
    side = Field()
    overdose = Field()


class DiseaseDescription(Item):
    """
    Заболевание: 
    - название
    - подробное описание
    - лекарства
    """
    url = Field()
    name = Field()
    description = Field()
    drugs = Field()
