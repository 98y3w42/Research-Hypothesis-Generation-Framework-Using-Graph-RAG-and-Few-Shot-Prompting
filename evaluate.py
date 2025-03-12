import os
import json
from openai import OpenAI
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from utils import instruction_prompts, llm_generation_while_loop, recover_generated_title_to_exact_version_of_title, load_dict_title_2_abstract, if_element_in_list_with_similarity_threshold

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
model_name = "gpt-4o-mini"

client = OpenAI()

def query_with_llm(query):
    messages=[{"role":"user","content":query}]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def evaluate_for_one_hypothesis(gene_hyp, gold_hyp, keypoints):
    prompts = instruction_prompts('eval_matched_score')
    full_prompt = prompts[0] + gene_hyp + prompts[1] + gold_hyp + prompts[2] + keypoints + prompts[3]
    structured_gene = query_with_llm(full_prompt)
    return structured_gene

#dict_bkg2note

def load_chem_annotation(chem_annotation_path):
    chem_annotation = pd.read_excel(chem_annotation_path, 'Overall')
    bkg_q = list(chem_annotation[chem_annotation.columns[6]])
    note = list(chem_annotation[chem_annotation.columns[18]])
    dict_bkg2note = {}
    for cur_b_id, cur_b in enumerate(bkg_q):
        dict_bkg2note[cur_b.strip().replace("\n", "")] = note[cur_b_id].strip()
    return dict_bkg2note

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ': '))

def is_list(v):
    return type(v) is list

if __name__ == '__main__':
    keypoints = load_chem_annotation("./data/chem_research_2024.xlsx")
    
    #evaluate_files = ['hypgened_oneshot.json', 'hypgened_threeshot.json', 'hypgened_fiveshot.json', 'hypgened_oneshot_graphrag.json', 'hypgened_threeshot_graphrag.json', 'hypgened_fiveshot_graphrag.json']
    #evaluate_files = ['hypgened_zeroshot.json', 'hypgened_graphrag.json']
    evaluate_files = ['hypgened_moosechem.json']
    model = SentenceTransformer('all-mpnet-base-v2') 
    for file in evaluate_files:
        hypgened = json.load(open(file, 'r', encoding='utf-8'))
        results = []
        scores = []
        cos_sim = []
        bar = tqdm(total=46)
        for hypo in hypgened:
            bar.update(1)
            if is_list(hypo['hypo_gened']):
                result_list = []
                for hypo_extracted in hypo['hypo_gened']:
                    result = {}
                    result['question'] = hypo['question']
                    evaluated_result = evaluate_for_one_hypothesis(hypo_extracted, hypo['hypothesis'], keypoints[hypo['question'].strip().replace("\n", "")])
                    match = re.search(r'Matched score: (\d+)', evaluated_result)
                    result['score'] = int(match.group(1))
                    scores.append(result['score'])
                    embeddings = model.encode([hypo_extracted, hypo['hypothesis']])
                    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
                    result['cosine'] = similarity.item()
                    cos_sim.append(result['cosine'])
                    result_list.append(result)
                results.append(result_list)
            else:
                result = {}
                result['question'] = hypo['question']
                evaluated_result = evaluate_for_one_hypothesis(hypo['hypo_gened'], hypo['hypothesis'], keypoints[hypo['question'].strip().replace("\n", "")])
                match = re.search(r'Matched score: (\d+)', evaluated_result)
                result['score'] = int(match.group(1))
                scores.append(result['score'])
                embeddings = model.encode([hypo['hypo_gened'], hypo['hypothesis']])
                similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
                result['cosine'] = similarity.item()
                cos_sim.append(result['cosine'])
                results.append(result)
        print(file + " average score: " + str(sum(scores) / len(scores)) + ", cosine similarity: " + str(sum(cos_sim) / len(cos_sim)))
        save_json(results, './data/evld_' + file)