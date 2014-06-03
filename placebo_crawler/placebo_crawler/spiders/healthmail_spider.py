# coding: utf-8

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import DrugDescription, DiseaseDescription
# from scrapy import log
from html2text import html2text 

import time

# НЕ ДОПИСАН
class DrugsSpider(Spider):
    """
    Crawler for http://www.health.mail.ru
    """

    name = 'healthmail_drugs'
    allowed_domains = ['health.mail.ru']
    # Раздел: "Лекарства группированы по направлению, на что они действуют
    # (пищеварительный тракт и обмен веществ, дерматология и прочее)"
    start_urls = ['http://health.mail.ru/drug/']
    name_domain = 'http://health.mail.ru' # название домена для относительных ссылок на сайте


    def p_between_id(self, n, sel):
        lineN = '//div[@class="text margin_bottom_30 js-text_adaptive"]//h2[%s]/following-sibling::p/text()'%str(n)
        N = sel.xpath(lineN).extract()
        lineN_1 = '//div[@class="text margin_bottom_30 js-text_adaptive"]//h2[%s]/preceding-sibling::p/text()'%str(n+1)
        N_1 = sel.xpath(lineN_1).extract()
        return [text for text in (set(N) & set(N_1))]


    def parse_drug(self, catalog_rubric_name, catalog_item_name, response):
        sel = Selector(response)
        drug_name = sel.xpath('//h1[@class="page-info__title"]/text()').extract[0]

        context = sel.xpath('//div[@class="column__air"]').extract()[0]

        classification = ''

        all_subheads = sel.xpath('//div[@class="text margin_bottom_30 js-text_adaptive"]//h2/text()').extract()

        print(all_subheads)
        description, usage, contra, side, overdose = '', '', '', '', ''
        for i, subhead in enumerate(all_subheads):
            print(i, subhead)
            n = i+1

            if subhead == u"Форма выпуска, состав и упаковка":
                description = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Дозировка" or subhead == u"Показания":
                usage = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Противопоказания":
                contra = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Побочные действия":
                side = ''.join(self.p_between_id(n, sel))
            elif subhead == u"Передозировка":
                overdose = ''.join(self.p_between_id(n, sel))
            yield DrugDescription(  url=response.url,
                                name=drug_name,
                                classification=classification,
                                description=description,
                                usage=usage,
                                contra=contra,
                                side=side,
                                overdose=overdose,
                                info=html2text(context),
                            )

    def parse_list_of_drugs(self, catalog_rubric_name, catalog_item_name, response):
        sel = Selector(response)
        url_drugs = name_domain + sel.xpath('//a[@class="entry__link link-holder"]//@href').extract()
        for url in url_drugs:
            root_url = name_domain + url
            print(root_url)
            yield Request(root_url, callback=self.parse_drug)

    def parse(self, response):
        sel = Selector(response)

        print("\n\nstart\n\n")#

        catalog_rubrics = sel.xpath('//div[@class="hidden hidden_small"]//div[@class="catalog__rubric"]')

        

        for catalog_rubric in catalog_rubrics:
            #print(sel.xpath('//div[@class="hidden hidden_small"]//div[@class="catalog__rubric"]//span[@class="catalog__rubric__title"]').extract()[2])
            catalog_rubric_name = catalog_rubric.xpath('//span[@class="catalog__rubric__title"]/text()').extract()[0]
            catalog_items = catalog_rubric.xpath('//div[@class="cataloc__rubric__items"]//a[@class="catalog__item"]')

            print("\n\n  catalog_rubric_name=" + catalog_rubric_name + "\n\n")#
            print(len(catalog_items))
            for catalog_item in catalog_items:
                catalog_item_name = catalog_items.xpath('//span[@class="catalog__item__title"]/text()').extract()[0]
                catalog_item_link = name_domain + catalog_items.xpath('//@href').extract()
                print(catalog_item_name + " " + catalog_item_name)
                yield Request(catalog_item_link, callback=self.parse_list_of_drugs)
