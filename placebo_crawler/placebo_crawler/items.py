# coding: utf-8

from scrapy.item import Item, Field


class DrugDescription(Item):
    """
    Полное описание лекарства:
    - классификация по болезням
    - применение
    - противопоказания
    - побочные действия
    - передозировка
    info - вся информация (без категорий)
    """
    url = Field()
    name = Field()
    classification = Field()
    description = Field()
    usage = Field()
    contra = Field()
    side = Field()
    overdose = Field()
    info = Field()


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
