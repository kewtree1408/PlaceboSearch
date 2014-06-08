# coding: utf-8
import codecs


#! /usr/bin/env python
# coding: utf-8

from twisted.internet import reactor
from scrapy.crawler import Crawler
from scrapy import log, signals
from placebo_crawler.spiders.drug_synonyms_spider import DrugSynonymsSpider
from scrapy.utils.project import get_project_settings


def setup_crawler(spider):
    settings = get_project_settings()
    crawler = Crawler(settings)
    crawler.configure()
    crawler.signals.connect(reactor.stop, signal=signals.spider_closed)
    crawler.crawl(spider)
    crawler.start()


def main():
    spider = DrugSynonymsSpider()
    log.start()
    setup_crawler(spider)
    reactor.run()

    items = []
    with codecs.open('drug_synonyms.txt', encoding='utf8', mode='r') as file:
        synonyms = []
        for line in file:
            if line.startswith('synonyms'):
                line = line[10:]
                synonyms = line.split('|')

            elif line.startswith('name'):
                name = line[6:]
                temp_list = [name]
                temp_list.extend(synonyms)
                items.append(temp_list)

    d = {}

    for line in items:
        for word in line:
            raw = list(line)
            raw.remove(word)
            d[word] = raw

    with codecs.open('drug_synonyms_dictionary.txt', encoding='utf8', mode='w') as file:
        for pair in d.items():
            s = pair[0].strip() + '|'
            for word in pair[1]:
                s += word.strip() + ','
            s += '\n'
            file.write(s)


if __name__ == '__main__':
    main()