#! /bin/sh
for spider_name in mailru_diseases mailru_drugs medi_drugs yaslovari_drugs rlsnet_drugs rlsnet_diseases
do
    curl http://localhost:6800/schedule.json -d project='default' -d spider=$spider_name
done
