import os
from neo4j import GraphDatabase
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

driver = GraphDatabase.driver(uri="URI", auth=("neo4j", "password"))
model = SentenceTransformer('all-mpnet-base-v2') 

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            
    def reset(self):
        with self.__driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
                
        
    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

def store_embeddings_in_neo4j():
    with driver.session() as session:
        query_all_nodes = """
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN DISTINCT n.name AS node_id
        """
        result = session.run(query_all_nodes)
        all_nodes = [record["node_id"] for record in result]
        bar = tqdm(total = len(all_nodes))
        for node_id in all_nodes:
            bar.update(1)
            embedding_vector = model.encode(node_id)  # numpy.ndarray
            embedding_list = embedding_vector.tolist()  # DBに保存するためPythonのlist型へ

            query_set_embedding = """
                MATCH (n {name: $node_id})
                SET n.embedding = $embedding
            """
            session.run(query_set_embedding, node_id=node_id, embedding=embedding_list)

    print("All node embeddings have been stored in Neo4j.")

if __name__ == "__main__":
    conn = Neo4jConnection(
        uri='URI', # ex. neo4j://localhost:7687
        user='neo4j', # ex. neo4j
        pwd='password' # ex. password
    )
    #file_name = "./data/kousatuyou.json"
    #file_name = "triples_mention_deleted.json"
    file_name = "triples.json"
    with open(file_name, "r", encoding='UTF-8') as f:
        triples = json.load(f)
    for triple in tqdm(triples):
        query = """
        MERGE (subject:Entity {name: $subject})
        MERGE (object:Entity {name: $object})
        WITH subject, object
        CALL apoc.create.relationship(subject, $relationship, {}, object) YIELD rel
        RETURN rel
        """
        conn.query(query, parameters={'subject': triple["head"], 'relationship': triple["relation"], 'object': triple["tail"]}) 
    store_embeddings_in_neo4j()