#! /usr/bin/env python
# coding: utf-8

from twisted.internet import reactor
from scrapy.crawler import Crawler
from scrapy import log, signals
from placebo_crawler.spiders.rlsnet_spiders import DiseaseSpider, DrugsSpider
from scrapy.utils.project import get_project_settings


def setup_crawler(spider):
    settings = get_project_settings()
    crawler = Crawler(settings)
    crawler.configure()
    crawler.signals.connect(reactor.stop, signal=signals.spider_closed)
    crawler.crawl(spider)
    crawler.start()


def main():
	disease = DiseaseSpider()
	drugs = DrugsSpider()
	log.start()
	for spider in (disease, drugs):
	    setup_crawler(spider)
	reactor.run()


if __name__ == '__main__':
	main()
