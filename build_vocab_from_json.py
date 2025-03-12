import os
import sys
import json
import re
from tqdm import tqdm


def load_vocab(filename):
    f = open(filename)
    voc2id = {}
    for line in f:
        line = line.strip()
        voc2id[line] = len(voc2id)
    return voc2id

def output_dict(ele2id, filename):
    num_ele = len(ele2id)
    id2ele = {v: k for k, v in ele2id.items()}
    f = open(filename, "w")
    for i in range(num_ele):
        f.write(id2ele[i] + "\n")
    f.close()

def add_word_in_Relation(relation_file, vocab):
    rel2id = load_vocab(relation_file)
    max_len = 0
    oov = set()
    for rel in rel2id:
        domain_list = [rel.lower()]
        tp_list = []
        for domain_str in domain_list:
            tp_list += domain_str.split("_")
        # print(tp_list)
        words = []
        for w_idx, w in enumerate(tp_list):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w != '' and w not in vocab:
                vocab[w] = len(vocab)
                oov.add(w)
                words.append(vocab[w])
        if len(words) > max_len:
            max_len = len(words)
    print("Max length:", max_len)
    print("Total", len(vocab))
    print("OOV relation", len(oov))
    return vocab

def process_text(input_text):
    cleaned_text = re.sub(r'[\?\(\)]', '', input_text)
    words = re.findall(r'[a-zA-Z0-9-]+', cleaned_text)
    return [word.lower() for word in words]

def add_word_in_question(inpath):
    vocab = {}
    for split in ["train", "dev", "test"]:
        infile = os.path.join(inpath, split + ".json")
        with open(infile) as f:
            for line in f:
                tp_obj = json.loads(line)
                tp_dep = tp_obj["question"]
                tokens = process_text(tp_dep)
                for word in tokens:
                    if word not in vocab:
                        vocab[word] = len(vocab)
        print(split, len(vocab))
    return vocab


question_vocab = add_word_in_question(os.getcwd())
print(question_vocab)
relation_file = "./data/relations.txt"
full_vocab = add_word_in_Relation(relation_file, question_vocab)
out_file = os.path.join(os.getcwd(), "/data/vocab.txt")
output_dict(full_vocab, out_file)
