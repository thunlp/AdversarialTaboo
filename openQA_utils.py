import json
from pprint import pprint
from nltk import word_tokenize
import math
from random import choice
from collections import defaultdict
import random
from collections import OrderedDict
from judge_tools import *

random.seed(6)

def read(filename):
    with open(filename) as f:
        return json.load(f)

def save(file, name):
    with open(name, 'w') as f:
        json.dump(file, f, indent=4)

def hit(pred_answer, target_mentions):
    pred_tokens = word_tokenize(pred_answer)
    for target_mention in target_mentions:
        if target_mention in pred_tokens:
            return True
    return False

def gen_games(all_turns, turns_per_game):
    assert len(all_turns) == 50
    random.shuffle(all_turns)
    turns_total = len(all_turns)
    games = []
    for i in range(turns_total // turns_per_game):
        games.append(all_turns[i * turns_per_game: (i + 1) * turns_per_game])
    return games


def get_exp_logits(pred):
    if 'start_logit' in pred:
        return math.exp(pred['start_logit'] + pred['end_logit'])
    else:
        return math.exp(pred['score'])


def show_result(defender_win, show_detail=False):
    if show_detail:
        result = sorted(defender_win.items(), key= lambda x: sum(x[1]))
        pprint(result)

    att_win = 0
    dfd_win = 0 
    tie = 0
    total = 0
    turns_cnt = 0
    for k, vs in defender_win.items():
        if k == 'turn_cnt':
            for v in vs:
                turns_cnt += v
        else:
            for v in vs: 
                total += 1
                if v == 0:
                    tie += 1
                elif v == 1:
                    dfd_win += 1
                elif v == -1:
                    att_win += 1
                else:
                    assert False, v

    result = OrderedDict(
        attacker_win = round(att_win / total, 4),
        defender_win = round(dfd_win / total, 4),
        tie = round(tie / total, 4),
        avg_turns = round(turns_cnt / total, 4),
        total = total
    )
    pprint(result)

def get_turn_data(node, word, word_turn_data, cp_turn_data, word_graph):
    if node == word:
        return choice(word_turn_data[node])
    
    random.shuffle(word_graph)
    for cp in word_graph:
        if cp in cp_turn_data:
            return choice(cp_turn_data[cp])
    
    return choice(word_turn_data[word])


def get_ans_diff_from_top(n_best_preds, honest_thresh):
    for pred in n_best_preds:
        if get_exp_logits(pred) < honest_thresh:
            return n_best_preds[0]['text'].lower()
        if n_best_preds[0]['text'].lower() not in pred['text'].lower():
            return pred['text'].lower()
    return n_best_preds[0]['text'].lower()


def choose_next_node(node, word, graph, return_prob):
    if graph[node] == []:
        return word
    
    if random.random() < return_prob:
        return word
    else:
        return choice(graph[node])

def get_config():
    config = {}
    config["turns_per_game"] = 10
    config["guess_limit"] = 1
    # We adopt a relatively loose criteria for the judge, 
    # since we are more focused on the game itself rather than the conversation quality 
    config["matching_thresh"] = -5
    config["ppl_thresh"] = 4000
    config["word_to_mention"] = './data/word_to_mentions_uncased.json'
    config["concept_graphs"] = './data/concept_graphs.json'


    config["BERT"] = {}
    config["BERT"]["guess_thresh"] = 10
    config["BERT"]["return_prob"] = 0.6
    config["BERT"]["word_conf_thresh"] = 3000
    config["BERT"]["honest_thresh"] = 1
    config["BERT"]["word_data"] = './data/BERT_word_data.json'
    config["BERT"]["concept_data"] = './data/BERT_concept_data.json'


    config["DocQA"] = {}
    config["DocQA"]["guess_thresh"] = 1000
    config["DocQA"]["return_prob"] = 0.6
    config["DocQA"]["word_conf_thresh"] = 1000000
    config["DocQA"]["honest_thresh"] = 100000
    config["DocQA"]["word_data"] = './data/DocQA_word_data.json'
    config["DocQA"]["concept_data"] = './data/DocQA_concept_data.json'

    return config