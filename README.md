# Research-Hypothesis-Generation-Framework-Using-Graph-RAG-and-Few-Shot-Prompting

## Overview
This framework leverages Large Language Models (LLMs) to generate research hypotheses in the field of chemistry. It utilizes data from the MOOSE-Chem framework (https://arxiv.org/abs/2410.07076).

Before running this framework, please place the following files in the `data` folder:
- `chem_research_2024.xlsx`
- `Inspiration_Corpus_3000.json`

Additionally, the following external tools are required:
- **Neo4j** (https://neo4j.com/): For managing the graph database
- **OpenAI API** (https://openai.com/index/openai-api/): For utilizing the language model

Before executing the program, ensure that `URI` and `OPENAI_API_KEY` are properly set within the code.

## Execution Steps
Below are the execution steps and roles of each script in this framework:

1. **`data_to_graph.py`**
   - Converts the MOOSE-Chem framework data into a graph format on Neo4j.
   - Saves the generated graph as `export.graphml` in the `data` folder.

2. **`kg_build.py`**
   - Extracts nodes and triples from `export.graphml`.

3. **`neo4j_triples.py`**
   - Adds embeddings to the extracted graph data.

4. **`xlsx_json.py`**
   - Extracts research topics and ground-truth research hypotheses from `chem_research_2024.xlsx`.

5. **`del_mentions.py`**
   - Removes unnecessary relations and cleans the data.

6. **`hypo_gen.py`**
   - Generates research hypotheses using Graph RAG (Graph Retrieval-Augmented Generation) and Few-Shot Prompting.

7. **`evaluate.py`**
   - Evaluates the generated research hypotheses.

8. **`analyze_evl.py`**
   - Analyzes the evaluation results.

9. **`plot.py`**
   - Visualizes the analyzed data.

## License
This project is licensed under the MIT License. See the LICENSE.txt file for details.

Copyright (c) 2025 Yusuke Saito

