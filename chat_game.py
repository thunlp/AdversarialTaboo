from collections import defaultdict
import random
from nltk import pos_tag
from chat_utils import *

random.seed(6)
config = get_config()

def get_strategy_data(model, strategy, mention_to_word):

    filename = config[model][strategy]

    if strategy == "golden_trigger":
        word_sents = {}
        data = read(filename)
        for word, turns in data.items():
            if word == '[NA]':
                continue
            assert len(turns) == 200
            for turn in turns:
                resp_org = set([mention_to_word.get(_x, _x) for _x in turn['response']])
                assert word in resp_org
            random.shuffle(turns)
            word_sents[word] = turns[:50]
        return word_sents

    if strategy == "neural_based":
        word_sents = {}
        data = read(filename)
        for word, turns in data.items():
            if word == '[NA]':
                continue
            assert len(turns) >= 50
            for turn in turns:
                resp_org = set([mention_to_word.get(_x, _x) for _x in turn['response']])
                assert word not in resp_org
            word_sents[word] = turns[:50]
        return word_sents

    if model == "ConceptFlow" and strategy == 'API_based':
        word_sents = {}
        data = read(filename)
        for word, turns in data.items():
            if word == '[NA]':
                continue
            if len(turns) < 50:
                turns.extend(data['thing'][: 50 - len(turns)])

            assert len(turns) == 50
            word_sents[word] = turns
        return word_sents
    
    if model == "DialoGPT" and strategy == 'API_based':
        word_sents = {}
        data = read(filename)
        for word, turns in data.items():
            if word == '[NA]':
                continue
            assert len(turns) == 50
            word_sents[word] = turns
        return word_sents


def get_DialoGPT_defense(turn, conf_hist):
    word_scores = turn['word_scores']
    scores = []
    tokens = []
    for i, x in enumerate(word_scores):
        if x['word'] == '':
            continue
        tokens.append(x['word'])
        scores.append(x['score'])
    pos_tag_result = pos_tag(tokens)

    assert len(tokens) == len(scores)
    for i in range(len(pos_tag_result)):
        mention, pos = pos_tag_result[i]
        if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and len(mention) > 2:
            conf_hist[mention] += float(scores[i])


def game(model, strategy, defender_guess):
    print("Running  Model: {}  Strategy: {}  Defender guess: {}".format(model, strategy, defender_guess))

    mention_to_word = read(config["mention_to_word"])
    word_turn_data = get_strategy_data(model, strategy, mention_to_word)

    result_record = {}
    result_record['turn_cnt'] = []
    for word, turns in word_turn_data.items():
        result_record[word] = []
        for game in gen_games(turns, config["turns_per_game"]):
            result = 0
            guess = 0
            turn_cnt = 0
            conf_hist = defaultdict(float)
            for turn in game:
                turn_cnt += 1
                # check if the utterance from attacker is fluent
                if turn['post_ppl'] > config["ppl_thresh"]:
                    result = 1
                    break
                
                if defender_guess:
                    if model == "DialoGPT":
                        get_DialoGPT_defense(turn, conf_hist)
                    else:
                        for x in turn['top5_keywords']:
                            conf_hist[x['word']] += float(x['score'])
                    
                    top_conf = sorted(conf_hist.items(), key=lambda x: x[1], reverse=True)
                    if len(top_conf) > 0:
                        guess_w, score = top_conf[0]
                        if score > config[model]["guess_thresh"] and guess < config["geuss_limit"]:
                            # guess the taboo word
                            guess += 1
                            if mention_to_word.get(guess_w, guess_w) == word:
                                result = 1
                                break
                # check if the utterance from defender is fluent and relevant
                if turn['inference'] == '' or turn['inf_ppl'] > config["ppl_thresh"] or turn['matching_score'] < config["matching_thresh"]:
                    result = -1
                    break
                resp_org = set([mention_to_word.get(_x, _x) for _x in turn['inference'].split()])
                # check if defender is attacked
                if word in resp_org:
                    result = -1
                    break
            result_record['turn_cnt'].append(turn_cnt)
            result_record[word].append(result)

    show_result(result_record)    

        

if __name__ == '__main__':
    # ConceptFlow experiments
    game(model="ConceptFlow", strategy="golden_trigger", defender_guess=False)
    game(model="ConceptFlow", strategy="golden_trigger", defender_guess=True)

    game(model="ConceptFlow", strategy="neural_based", defender_guess=False)
    game(model="ConceptFlow", strategy="neural_based", defender_guess=True)

    game(model="ConceptFlow", strategy="API_based", defender_guess=False)
    game(model="ConceptFlow", strategy="API_based", defender_guess=True)

    # DialoDialoGPT experiments
    game(model="DialoGPT", strategy="golden_trigger", defender_guess=False)
    game(model="DialoGPT", strategy="golden_trigger", defender_guess=True)

    game(model="DialoGPT", strategy="neural_based", defender_guess=False)
    game(model="DialoGPT", strategy="neural_based", defender_guess=True)

    game(model="DialoGPT", strategy="API_based", defender_guess=False)
    game(model="DialoGPT", strategy="API_based", defender_guess=True)




