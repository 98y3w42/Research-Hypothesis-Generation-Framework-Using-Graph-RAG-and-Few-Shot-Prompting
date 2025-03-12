import json

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ': '))

def judge_graph_density_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = set()
    edges = set()
    
    for triple in data:
        head = triple["head"]
        relation = triple["relation"]
        tail = triple["tail"]
        
        # ノード集合を更新
        nodes.add(head)
        nodes.add(tail)
        edges.add(relation)
    
    n = len(nodes)  # ノード数
    
    print("ノード数:" + str(n))
    print("エッジ数:" + str(len(data)))
    print("リレーション数" + str(len(edges)))

    # ノード数が 0 または 1 の場合は、エッジが成り立たない
    if n < 2:
        print("頂点数が 1 未満なので、疎密は判定できません。")
        return
    
    # 有向グラフの場合、自己ループを除く最大エッジ数は n*(n-1)
    max_edges = n * (n - 1)
    
    # 密度を計算
    density = len(edges) / max_edges
    return density

if __name__ == "__main__":
    with open("./data/triples.json", "r", encoding='UTF-8') as f:
        triples = json.load(f)
    deleted_triples = []
    for triple in triples:
        if triple["relation"] != "MENTIONS":
            deleted_triples.append(triple)
    print(len(deleted_triples))
    save_json(deleted_triples, "./data/triples.json")
    
    #疎か密か
    print(judge_graph_density_from_json("./data/triples.json"))