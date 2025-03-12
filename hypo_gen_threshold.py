import os
import sys
import time  # 実行時間計測用
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings import OpenAIEmbeddings
from neo4j import GraphDatabase

from openai import OpenAI
import tiktoken

import arxiv
import re
from pypdf import PdfReader
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

from typing import Optional
import faiss
import os
import json
from openai import OpenAI
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import statistics
from evaluate import load_chem_annotation

from utils import instruction_prompts, llm_generation_while_loop, recover_generated_title_to_exact_version_of_title, load_dict_title_2_abstract, if_element_in_list_with_similarity_threshold


os.environ["NEO4J_URI"] = "URI"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
model_name = "gpt-4o-mini"

model = SentenceTransformer('all-mpnet-base-v2')

llm = ChatOpenAI(model_name=model_name, temperature=0)
client = OpenAI()
driver = GraphDatabase.driver(uri="bolt://localhost:7687", auth=("neo4j", "password"))

# 1) すべてのノード埋め込みをNeo4jから取得し、一括でfaissに登録
with driver.session() as session:
    result = session.run("""
        MATCH (n) 
        WHERE n.name IS NOT NULL AND n.embedding IS NOT NULL
        RETURN n.name AS node_id, n.embedding AS embedding
    """)
    node_ids = []
    embeddings = []
    for record in result:
        node_ids.append(record["node_id"])
        embeddings.append(record["embedding"])
    
    embeddings = np.array(embeddings, dtype=np.float32)
    d = embeddings.shape[1]  # 次元数
# 2) インデックス作成（例: L2距離 + IVFインデックス）
index = faiss.index_factory(d, "IVF100,Flat", faiss.METRIC_L2)
index.train(embeddings)
index.add(embeddings)

# node_idsを後で逆引きできるように保持
id_to_node = {i: node_ids[i] for i in range(len(node_ids))}

enc = tiktoken.encoding_for_model(model_name)
    
def query_with_llm(query):
    messages=[{"role":"user","content":query}]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def find_similar_node_and_edges_ANN(query_text, threshold=0.8, top_k=5):
    """ANNで近似近傍検索を行い、最も近いノードとエッジを取得"""
    query_emb = model.encode(query_text).astype(np.float32)[None, :]  # shape(1, d)
    
    distances, indexes = index.search(query_emb, top_k)
    
    best_node_id = None
    best_similarity = -1.0
    for dist, idx in zip(distances[0], indexes[0]):
        if idx < 0:  # faissが見つからない場合に -1 が返ることがある
            continue
        candidate_node = id_to_node[idx]
        
        # 改めてコサイン類似度を計算
        candidate_emb = embeddings[idx]
        dot = np.dot(query_emb[0], candidate_emb)
        norm_q = np.linalg.norm(query_emb[0])
        norm_c = np.linalg.norm(candidate_emb)
        cos_sim = dot / (norm_q * norm_c + 1e-10)
        
        if cos_sim > best_similarity:
            best_similarity = cos_sim
            best_node_id = candidate_node
    
    # 3) 閾値チェック
    if best_node_id is None or best_similarity < threshold:
        return None
    
    # 4) Neo4jからエッジ情報を取得
    with driver.session() as session:
        edges_info = []
        edge_results = session.run("""
            MATCH (n {name: $best_node_id})-[r]-(m)
            RETURN n, r, m
        """, best_node_id=best_node_id)
        
        for record in edge_results:
            n_node = record["n"]
            r_rel = record["r"]
            m_node = record["m"]
            edges_info.append({
                "head": n_node.get("name"),
                "relation": r_rel.type,
                "tail": m_node.get("name")
            })
        edge_results = session.run("""
            MATCH (n)-[r]-(m {name: $best_node_id})
            RETURN n, r, m
        """, best_node_id=best_node_id)
        
        for record in edge_results:
            n_node = record["n"]
            r_rel = record["r"]
            m_node = record["m"]
            edges_info.append({
                "head": n_node.get("name"),
                "relation": r_rel.type,
                "tail": m_node.get("name")
            })

    return edges_info

def hypgen_with_graph_rag(qa, fs, threshold=0.8):
    answers = []
    bar = tqdm(total=len(qa))  
    tokens = 0
    for temp in qa:
        bar.update(1)
        answer = {}
        question_text = temp['question']
        nodes = get_nodes_from_question(question_text)
        edges = []
        for node_id in nodes:
            edges_info = find_similar_node_and_edges_ANN(node_id, threshold=threshold)
            if edges_info:
                edges.extend(edges_info)
        
        few_edges = ''.join(["Edge" + str(i+1) + ": " + json.dumps(edges[i]) + "\n" for i in range(len(edges))])
        few_exp = ''.join(["Example" + str(i+1) + ": " + json.dumps(fs[i]) + "\n" for i in range(len(fs))])
        
        prompts = [
            "You are helping to develop Chemistry research hypotheses based on few related knowledge graph edges and few examples. A senior researcher has identified the research question, and few related knowledge graph edges and few examples are provided, each containing a specific research question and its corresponding hypothesis. These examples offer insights into different aspects of the main research question. Your task is to analyze these related knowledge graph edges and generate a new hypothesis that integrates the key insights from few related knowledge graph edges and few examples, ensuring the hypothesis demonstrates Validness, Novelty, and Significance, as is typical of papers published in <Nature> or <Science>. \nThe main research question is:",
            question_text,
            "\n\nThe edges are provided below in the following format:\n"
            "- Edge1: {'head': '...', 'relation': '...', 'tail': '...'}\n"
            "- Edge2: {'head': '...', 'relation': '...', 'tail': '...'}\n",
            few_edges,
            "\n\nThe examples are provided below in the following format:\n- Example 1: {'question': '<Research question>', 'hypothesis': '<Hypothesis>'}\n- Example 2: {'question': '<Research question>', 'hypothesis': '<Hypothesis>'}\n- Example 3: {'question': '<Research question>', 'hypothesis': '<Hypothesis>'}\n- Example 4: {'question': '<Research question>', 'hypothesis': '<Hypothesis>'}\n- Example 5: {'question': '<Research question>', 'hypothesis': '<Hypothesis>'}\n",
            few_exp,
            "\nNow you have seen the main question and these edges. Please try to generate a new answer that integrates the key insights from these edges. Your response should include a clear hypothesis and the reasoning process. (response format): \n{{\n'Hypothesis': \n'Reasoning Process':\n}}"
        ]
        prompt = '\n'.join(prompts)
        result = query_with_llm(prompt)
        tokens += len(enc.encode(prompt))
        
        match = re.search(r"'Hypothesis':\s*'(.*?)',", result)
        
        answer['question'] = question_text
        answer['hypothesis'] = temp['hypothesis']
        if not match:
            match = re.search(r"\"Hypothesis\":\s*\"(.*?)\",", result)
        if match:
            answer['hypo_gened'] = match.group(1)
        else:
            answer['hypo_gened'] = result

        answers.append(answer)
    print(tokens)
    return answers

def get_nodes_from_question(question):
    transformer = LLMGraphTransformer(llm=llm)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    documents = text_splitter.create_documents([question])
    docs = text_splitter.split_documents(documents)
    graph_documents = transformer.convert_to_graph_documents(docs)
    node_ids = []
    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            node_ids.append(node.id)
    return node_ids
        
def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    sys.stdout = open(str(timestamp) + "_hypo_gen_sikiichi.log", "w")
    print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    moose_file = "./data/moosechem_qa.json"
    moose_fs_file = "./data/moosechem_qa.json"
    moose_qa = json.load(open(moose_file, 'r', encoding='utf-8'))
    moose_fs = json.load(open(moose_fs_file, 'r', encoding='utf-8'))
    keypoints = load_chem_annotation("./data/chem_research_2024.xlsx")
    
    model_hikaku = SentenceTransformer('all-mpnet-base-v2') 
    # hypgen_with_graph_rag (5-shot)
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        hypgened = hypgen_with_graph_rag(moose_qa, moose_fs, threshold)
        bar = tqdm(total=46)
        cosines = []
        for hypo in hypgened:
            bar.update(1)
            embeddings_hikaku = model_hikaku.encode([hypo['hypo_gened'], hypo['hypothesis']])
            similarity = util.pytorch_cos_sim(embeddings_hikaku[0], embeddings_hikaku[1])
            cosines.append(similarity.item())
            
        print(threshold, sum(cosines) / len(cosines), statistics.pvariance(cosines))
    
    print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    sys.stdout.close()
    sys.stdout = sys.__stdout__
