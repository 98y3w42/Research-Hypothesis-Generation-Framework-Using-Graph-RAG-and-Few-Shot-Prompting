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

os.environ["NEO4J_URI"] = "URI"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
os.environ["OPENAI_API_KEY"] = "OPEN_AI_KEY"
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
            MATCH (n {name: $best_node_id})-[r]->(m)
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
            MATCH (n)-[r]->(m {name: $best_node_id})
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

def get_triples(question):
    nodes = get_nodes_from_question(question)
    edges = []
    for node_id in nodes:
        edges_info = find_similar_node_and_edges_ANN(node_id)
        if edges_info:
            edges.extend(edges_info)
    return edges

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
    question = "How can CO2 be efficiently converted into valuable chemicals, such as formic acid, using an electrolysis process that avoids carbonate precipitation and ensures long-term operational stability?"
    data = get_triples(question)
    save_json(data, "./data/kousatuyou.json")
