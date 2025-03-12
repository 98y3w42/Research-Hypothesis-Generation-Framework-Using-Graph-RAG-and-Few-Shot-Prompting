import re
import xml.etree.ElementTree as ET
import json
import os

def get_nodes(xml_string, corpus):
    namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    root = ET.fromstring(xml_string)
    result = {}

    for node in root.findall(".//graphml:node", namespace):
        node_id = node.get("id")  # nodeのid属性を取得
        node_text = None

        # graphml:data要素を取得し、textまたはidの内容を判定
        for data in node.findall("graphml:data", namespace):
            key = data.get("key")
            if key == "text":  # Document用のtextフィールド
                flag = True
                for paper in corpus:
                    if data.text in paper[1]:
                        node_text = paper[1]
                        flag = False
                        break
                if flag:
                    node_text = data.text
            elif key == "id" and node_text is None:  # Entity用のidフィールド
                node_text = data.text

        # データをリストに追加
        if node_id and node_text:
            result[node_id] = node_text

    return result

def get_triples(xml_string, nodes):
    namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    root = ET.fromstring(xml_string)
    triples = []

    for edge in root.findall(".//graphml:edge", namespace):
        triple_id = edge.get("id")  # nodeのid属性を取得
        if triple_id:
            triple = {}
            triple["head"] = nodes[edge.get("source")].rstrip().replace("\n", "")
            triple["relation"] = edge.get("label")
            triple["tail"] = nodes[edge.get("target")].rstrip().replace("\n", "")
            triples.append(triple)

    return triples

def get_json(dir):
    temp = None
    with open(dir, 'r') as f:
        temp = json.load(f)
    return temp

if __name__ == '__main__':
    xml_texts = ""
    with open('./data/export.graphml', encoding='utf-8') as f:
        xml_texts = f.read()
    
    #略されているドキュメントを、略していない形に変換
    corpus_dir = os.getcwd() + "\\data\\Inspiration_Corpus_3000.json"
    corpus = get_json(corpus_dir)
    nodes = get_nodes(xml_texts, corpus)
    triples = get_triples(xml_texts, nodes)
    print("Nodes: " + str(len(nodes)))
    print("Triples: " + str(len(triples)))
    with open('nodes.json', 'w') as f:
        json.dump(nodes, f)
    with open('triples.json', 'w') as f:
        json.dump(triples, f)
        