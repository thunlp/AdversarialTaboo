import json
from pprint import pprint
from nltk import word_tokenize
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

def gen_games(all_turns, turns_per_game):
    assert len(all_turns) == 50
    random.shuffle(all_turns)
    turns_total = len(all_turns)
    games = []
    for i in range(turns_total // turns_per_game):
        games.append(all_turns[i * turns_per_game: (i + 1) * turns_per_game])
    return games


def get_config():
    config = {}
    config["turns_per_game"] = 10  
    config["geuss_limit"] = 1
    # We adopt a relatively loose criteria for the judge, 
    # since we are more focused on the game itself rather than the conversation quality 
    config["ppl_thresh"] = 50000
    config["matching_thresh"] = -10
    config["mention_to_word"] = 'data/mention_to_word_uncased.json'


    config["DialoGPT"] = {}
    config["DialoGPT"]["guess_thresh"] = 0.03
    config["DialoGPT"]["golden_trigger"] = 'data/DialoGPT_golden_trigger.json'
    config["DialoGPT"]["neural_based"] = 'data/DialoGPT_neural_based.json'
    config["DialoGPT"]["API_based"] = 'data/DialoGPT_API_based.json'

    config["ConceptFlow"] = {}
    config["ConceptFlow"]["guess_thresh"] = 0.1
    config["ConceptFlow"]["golden_trigger"] = 'data/ConceptFlow_golden_trigger.json'
    config["ConceptFlow"]["neural_based"] = 'data/ConceptFlow_neural_based.json'
    config["ConceptFlow"]["API_based"] = "data/ConceptFlow_API_based.json"

    return config
