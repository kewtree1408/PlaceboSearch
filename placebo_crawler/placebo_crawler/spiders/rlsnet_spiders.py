# coding: utf-8

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.spider import Spider
from scrapy.http import Request
from placebo_crawler.items import Link, DrugDescription, DrugInfo, DiseaseDescription
# from scrapy import log
from html2text import html2text 

import time


class LecTreeSpider(Spider):
    """
    Crawler for http://www.rlsnet.ru/lec_tree.htm
    """

    name = 'rlsnet_lectree'
    allowed_domains = ['rlsnet.ru']
    # Раздел "Лекарственные средства"
    start_urls = ['http://www.rlsnet.ru/lec_index_id_1.htm']

    # выдираем p-ку строго после заданного заголовка (скорее всего работает не для всех страниц)
    # todo: переделать так, чтобы работало для всех страниц
    def p_between_h(self, number_hN, sel):
        # lineN = '//h2[%s]/following-sibling::p/text()'%str(number_hN)
        lineN = '//h2[%s]/following-sibling::p/text()'%str(number_hN)
        hN = sel.xpath(lineN).extract()
        # lineN_1 = '//h2[%s]/preceding-sibling::p/text()'%str(number_hN+1)
        lineN_1 = '//h2[%s]/preceding-sibling::p/text()'%str(number_hN+1)
        hN_1 = sel.xpath(lineN_1).extract()
        return [text for text in (set(hN) & set(hN_1))]

    def parse_disease(self, response):
        sel = Selector(response)
        disease_title = sel.xpath('//h1[@id="page_head"]/text()').extract()[0]
        # context = sel.xpath('//*[@id="div_nest"]').extract()[0]
        context = sel.xpath('//*[@class="news_text full"]').extract()[0]
        drugs_and_stuff = ''.join([stuff+'\n' for stuff in sel.xpath('//a[@class="rest_data_list"]/text()').extract()])
        yield DiseaseDescription(   url = response.url,
                                    name = disease_title, 
                                    description = html2text(context), 
                                    drugs = drugs_and_stuff,
                                )

    def parse_drug(self, response):
        sel = Selector(response)
        drug_name = sel.xpath('//span[@part="rusname"]/text()').extract()[0]
        context = sel.xpath('//*[@id="tn_content"]').extract()[0]
        # сохраняем всю информацию для индекса
        time.sleep(1)
        yield DrugInfo(url=response.url, name=drug_name, info=html2text(context))

        # сохраняем всю информацию для сниппета (пока не делаем, выяснить, нужно ли делать?)
        description = ''.join(self.p_between_h(6, sel))
        usage = ''.join(self.p_between_h(9, sel))
        contra = ''.join(self.p_between_h(10, sel))
        side = ''.join(self.p_between_h(11, sel))
        overdose = ''.join(self.p_between_h(12, sel))
        time.sleep(1)
        yield DrugDescription(  url = response.url,
                                name = drug_name,
                                description = description,
                                usage = usage,
                                contra = contra,
                                side = side,
                                overdose = overdose,
                            )

        classification = u''
        for nozo_class in sel.xpath('//*[@class="field_20110303"]/li/a'):
            nozo_title = nozo_class.xpath('text()').extract()[0]
            classification += nozo_title
            nozo_url = nozo_class.xpath('@href').extract()[0]
            time.sleep(1)
            yield Request(nozo_url, callback=self.parse_disease)

         
    def parse_item(self, response):
        sel = Selector(response)
        for rest_data in sel.xpath('//*[@class="rest_data"]/a'):
            name = rest_data.xpath('text()').extract()[0]
            url = rest_data.xpath('@href').extract()[0]
            time.sleep(2)
            yield Request(url, callback=self.parse_drug)
            break

    def parse(self, response):
        sel = Selector(response)
        for drug_form in sel.xpath('//*[@class="subcatlist"]/ul/li/a'):
            url = drug_form.xpath('@href').extract()[0]
            yield Request(url, callback=self.parse_item)
            print "AAAAAAAA1111"
            break
            time.sleep(2)
        print "AAAAAAAA2222"

        sel = Selector(response)
        print sel
        for drug in sel.xpath('//*[@class="subcatlist"]/ul/li/a'):
            print "AAAAAAA3333"
            url = drug.xpath('@href').extract()[0]
            yield Request(url, callback=self.parse) 
            time.sleep(2)
            break
            
            

# class LecTreeSpider1(CrawlSpider):
#     """
#     Crawler for http://www.rlsnet.ru/lec_tree.htm
#     """

#     name = 'rlsnet_lectree'
#     allowed_domains = ['rlsnet.ru']
#     start_urls = ['http://www.rlsnet.ru']

#     # Пока полезное нам только в разделе "Лекарственные средства"
#     rules = (
#              # Rule(SgmlLinkExtractor(allow=('search\.php\?.+')), follow=True),
#              Rule(SgmlLinkExtractor(allow=('lec_index_id_\[0-9]+.htm')), callback='parse_item'),
#             )

#     def parse_item(self, response):
#         sel = Selector(response)
#         for drug_form in sel.xpath('//*[@class="subcatlist"]/ul/li'):
#             name = drug_form.xpath('a/text()').extract()
#             url = drug_form.xpath('a/@href').extract()
#             yield Link(name=name, url=url)

# class RlsnetSpider(Spider):

#     name = 'rlsnet.ru'
#     allowed_domains = ['rlsnet.ru']
#     # see: http://www.rlsnet.ru/robots.txt
#     start_urls = [ 'http://www.rlsnet.ru' + 
#                     allowed_url for allowed_url in (
#                                 '/baa_fg_id_370.htm',
#                                 '/baa_fg_id_372.htm',
#                                 '/baa_fg_id_363.htm',
#                                 '/baa_fg_id_361.htm',
#                                 '/baa_fg_id_366.htm',
#                                 '/baa_fg_id_359.htm',
#                                 '/baa_fg_id_362.htm',
#                                 '/baa_fg_id_365.htm',
#                                 '/baa_fg_id_367.htm',
#                                 '/baa_fg_id_360.htm',
#                                 '/baa_fg_id_368.htm',
#                                 '/baa_fg_id_369.htm',
#                                 '/baa_fg_id_595.htm',
#                                 '/baa_fg_id_371.htm',
#                     )
#                 ]

#     def parse(self, response):
#         sel = Selector(response)
        

