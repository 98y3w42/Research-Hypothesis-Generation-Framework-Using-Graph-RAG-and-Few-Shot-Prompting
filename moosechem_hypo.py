import os
import json

        
def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    moose_file = "./data/moosechem_qa.json"
    moose_qa = json.load(open(moose_file, 'r', encoding='utf-8'))
    moose_dict = "./data/evaluated_moose_chem/"
    moose_default = "evaluation_gpt4_corpus_300_survey_0_gdthInsp_1_intraEA_1_interEA_1_bkgid_"
    output = "./data/hypgened_moosechem.json"
    results = []
    for i in range(46):
        result = {}
        moose_evaluated = moose_dict + moose_default + str(i + 5) + ".json"
        result['question'] = moose_qa[i]['question']
        result['hypothesis'] = moose_qa[i]['hypothesis']
        hypo_gened = []
        with open(moose_evaluated, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for hyp in data[-1]:
                hypo_gened.append(hyp[0])
        result['hypo_gened'] = hypo_gened
        results.append(result)

    save_json(results, output)
    
