# coding: utf-8

from scrapy.item import Item, Field

class DrugInfo(Item):
    """
    Вся информация на странице (без категорий)
    name - название лекарства
    info - вся информация
    """
    name = Field()
    info = Field()
    url = Field()



class DrugDescription(Item):
    """
    Полное описание лекарства:
    - классификация по болезням
    - применение
    - противопоказания
    - побочные действия
    - передозировка
    """
    name = Field()
    classification = Field()
    description = Field()
    usage = Field()
    contra = Field()
    side = Field()
    overdose = Field()
    url = Field()


class DiseaseDescription(Item):
    """
    Заболевание: 
    - название
    - подробное описание
    - лекарства
    """
    name = Field()
    description = Field()
    drugs = Field()
    url = Field()



class Link(Item):
    """
    Пара: название - ссылка
    """
    name = Field()
    url = Field()
