from collections import defaultdict
import random
from random import choice
from openQA_utils import *
from judge_tools import *

random.seed(6)
config = get_config()

def stage_1(model):
    word_to_mention = read(config["word_to_mention"])
    word_turn_data = read(config[model]["word_data"])

    result_record = {}
    result_record['turn_cnt'] = []
    for word, turns in word_turn_data.items():

        result_record[word] = []
        # generate games for each taboo word
        for game in gen_games(turns, config["turns_per_game"]):
            result = 0
            turn_cnt = 0
            for turn in game:
                turn_cnt += 1                    
                pred_answer = turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['text'].lower()
                # check if the question is fluent
                if turn['q_ppl'] > config["ppl_thresh"]:
                    result = 1
                    break
                # check if the answer is relevant
                if turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['matching_score'] < config["matching_thresh"]:
                    result = -1
                    break
                # check if the defender is attacked
                if hit(pred_answer, word_to_mention[word]):
                    result = -1
                    break
            result_record['turn_cnt'].append(turn_cnt)
            result_record[word].append(result)
    
    print("Stage 1: {}".format(model))
    show_result(result_record)
    
                
def stage_2(model):
    word_to_mention = read(config["word_to_mention"])
    word_turn_data = read(config[model]["word_data"])

    result_record = {}
    result_record['turn_cnt'] = []
    for word, turns in word_turn_data.items():

        result_record[word] = []
        for game in gen_games(turns, config["turns_per_game"]):
            result = 0
            guess = 0
            turn_cnt = 0
            for turn in game:
                turn_cnt += 1
                top_pred_answer = turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['text'].lower()
                # get confidence of the answer
                top_pred_elogit = get_exp_logits(turn['paragraphs'][0]['qas'][0]['n_best_preds'][0])
                if turn['q_ppl'] > config["ppl_thresh"]:
                    result = 1
                    break
                if len(word_tokenize(top_pred_answer)) == 1 and top_pred_elogit > config[model]["guess_thresh"] and guess < config["guess_limit"]:
                    guess += 1
                    if hit(top_pred_answer, word_to_mention[word]):
                        result = 1
                        break  
                    if turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['matching_score'] < config["matching_thresh"]:
                        result = -1
                        break
                else:
                    if turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['matching_score'] < config["matching_thresh"]:
                        result = -1
                        break
                    if hit(top_pred_answer, word_to_mention[word]):
                        result = -1
                        break
            result_record['turn_cnt'].append(turn_cnt)
            result_record[word].append(result)

    print("Stage 2: {}".format(model))
    show_result(result_record)


def stage_3(model):
    graphs = read(config["concept_graphs"])
    word_to_mention = read(config["word_to_mention"])
    word_turn_data = read(config[model]["word_data"])
    cp_turn_data = read(config[model]["concept_data"])

    result_record = {}
    result_record['turn_cnt'] = []
    for word in word_turn_data:
        graph = graphs[word]
        result_record[word] = []
        for _i in range(5):
            result = 0
            guess = 0
            node = choice(graph[word])
            turn_cnt = 0
            for _j in range(config["turns_per_game"]):
                turn_cnt += 1
                turn = get_turn_data(node, word, word_turn_data, cp_turn_data, graph[word])
                if turn['q_ppl'] > config["ppl_thresh"]:
                    result = 1
                    break
                top_pred_answer = turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['text'].lower()
                top_pred_elogit = get_exp_logits(turn['paragraphs'][0]['qas'][0]['n_best_preds'][0])
                if len(word_tokenize(top_pred_answer)) == 1 and top_pred_elogit > config[model]["guess_thresh"] and guess < config["guess_limit"]:
                    guess += 1
                    if hit(top_pred_answer, word_to_mention[word]):
                        result = 1
                        break  
                    if turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['matching_score'] < config["matching_thresh"]:
                        result = -1
                        break
                else:
                    if turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['matching_score'] < config["matching_thresh"]:
                        result = -1
                        break
                    if hit(top_pred_answer, word_to_mention[word]):
                        result = -1
                        break
                    
                node = choose_next_node(node=node, word=word, graph=graph, return_prob=config[model]["return_prob"])
            result_record['turn_cnt'].append(turn_cnt)
            result_record[word].append(result)

    print("Stage 3: {}".format(model))
    show_result(result_record)


def stage_4(model):
    graphs = read(config["concept_graphs"])
    word_to_mention = read(config["word_to_mention"])
    word_turn_data = read(config[model]["word_data"])
    cp_turn_data = read(config[model]["concept_data"])

    result_record = {}
    result_record['turn_cnt'] = []
    for word in word_turn_data:
        graph = graphs[word]
        result_record[word] = []
        # 5 games per word
        for _i in range(5):
            result = 0
            guess = 0
            node = choice(graph[word])
            word_conf = defaultdict(float)
            turn_cnt = 0

            for _j in range(config["turns_per_game"]):
                turn_cnt += 1
                turn = get_turn_data(node, word, word_turn_data, cp_turn_data, graph[word])
                if turn['q_ppl'] > config["ppl_thresh"]:
                    result = 1
                    break

                top_pred_answer = turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['text'].lower()

                top_pred_elogit = get_exp_logits(turn['paragraphs'][0]['qas'][0]['n_best_preds'][0])
                word_conf[top_pred_answer] += top_pred_elogit

                if len(word_tokenize(top_pred_answer)) == 1 and top_pred_elogit > config[model]["guess_thresh"] and guess < config["guess_limit"] and word_conf[top_pred_answer] > config[model]["word_conf_thresh"]:
                    guess += 1
                    # print("Guess:", top_pred_answer)
                    if hit(top_pred_answer, word_to_mention[word]):
                        result = 1
                        # print("Guess Correct!")
                        break  
                    # print("Guess Wrong. Game Continues.")
                    if turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['matching_score'] < config["matching_thresh"]:
                        result = -1
                        break
                else:
                    resp = get_ans_diff_from_top(turn['paragraphs'][0]['qas'][0]['n_best_preds'], config[model]["honest_thresh"])
                    # print("Defender:", resp)
                    if turn['paragraphs'][0]['qas'][0]['n_best_preds'][0]['matching_score'] < config["matching_thresh"]:
                        result = -1
                        break
                    if hit(resp, word_to_mention[word]):
                        result = -1
                        # print("Attacker Win.")
                        break
                node = choose_next_node(node=node, word=word, graph=graph, return_prob=config[model]["return_prob"])

            result_record['turn_cnt'].append(turn_cnt)
            result_record[word].append(result)

    print("Stage 4: {}".format(model))
    show_result(result_record)
    

if __name__ == '__main__':

    stage_1(model="BERT")
    stage_1(model="DocQA")

    stage_2(model="BERT")
    stage_2(model="DocQA")

    stage_3(model="BERT")
    stage_3(model="DocQA")

    stage_4(model="BERT")
    stage_4(model="DocQA")








    

    
