# coding: utf-8
import codecs


def main():
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