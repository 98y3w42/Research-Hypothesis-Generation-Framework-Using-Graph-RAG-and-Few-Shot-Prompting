# Graph Retrieval-Augmented Generation および Few-shot プロンプティングを用いた研究仮説生成フレームワーク

## 概要
本フレームワークは、大規模言語モデル（LLM）を活用して化学分野の研究仮説を生成するフレームワークです。データとして、MOOSE-Chemフレームワーク（https://arxiv.org/abs/2410.07076）で使用されたデータを利用します。

本フレームワークを実行する前に、`data` フォルダ内に以下のファイルを配置してください。
- `chem_research_2024.xlsx`
- `Inspiration_Corpus_3000.json`

また、以下の外部ツールが必要です。
- **Neo4j**（https://neo4j.com/）: グラフデータベースの管理
- **OpenAI API**（https://openai.com/index/openai-api/）: 言語モデルの利用

プログラム実行前に、プログラム内の `URI` および `OPENAI_API_KEY` を適切な形式で設定してください。

## 実行手順
本フレームワークの実行手順と各スクリプトの役割を以下に示します。

1. **`data_to_graph.py`**
   - MOOSE-ChemフレームワークのデータをNeo4j上のグラフに変換します。
   - 作成したグラフを `export.graphml` という名前で `data` フォルダ内に保存してください。

2. **`kg_build.py`**
   - `export.graphml` からノードとトリプルを抽出します。

3. **`neo4j_triples.py`**
   - 抽出したグラフデータに埋め込み（embedding）を付与します。

4. **`xlsx_json.py`**
   - `chem_research_2024.xlsx` から研究課題と研究仮説の正解データを抽出します。

5. **`del_mentions.py`**
   - 不要なリレーションを削除し、データを整理します。

6. **`hypo_gen.py`**
   - Graph RAG（Graph Retrieval-Augmented Generation）とFew-shotプロンプティングを用いて研究仮説を生成します。

7. **`evaluate.py`**
   - 生成した研究仮説の評価を行います。

8. **`analyze_evl.py`**
   - 評価結果を解析します。

9. **`plot.py`**
   - 解析後のデータを可視化します。

## License
This project is licensed under the MIT License, see the LICENSE.txt file for details

Copyright (c) 2025 Yusuke Saito