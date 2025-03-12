import json
import os

if __name__ == '__main__':
    tri_dir = '\\data\\triples.json'
    triples = json.load(open(os.getcwd() + tri_dir, encoding='utf-8'))
    
    entities = set()
    relations = set()
    for triple in triples:
        entities.add(triple["head"])
        relations.add(triple["relation"])
        entities.add(triple["tail"])
    with open('entities.txt', 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(entity.replace("\n", "") + "\n")
    with open('relations.txt', 'w', encoding='utf-8') as f:
        for relation in relations:
            f.write(relation.replace("\n", "") + "\n")