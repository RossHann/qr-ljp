import thulac
import torch
import os
import torch.nn as nn
import pickle as pk
from parameters import parse
from file_operations import get_json_content, get_jsonl_content, set_json_file
from string import punctuation
from pretrained_model.models import NumberEncoder

args = parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices


number_encoder = NumberEncoder(args)
if torch.cuda.is_available():
    number_encoder = nn.DataParallel(number_encoder)
    number_encoder.cuda()
number_encoder.load_state_dict(torch.load("./pretrained_model/number_encoder"))


add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc

cutter = thulac.thulac(seg_only=True)

article2id = get_json_content("./data/additional_data/article2id.json")
charge2id = get_json_content("./data/additional_data/charge2id.json")
time2id = get_json_content("./data/additional_data/time2id.json")
word2id = get_json_content("./data/additional_data/word2id_dict.json")


def get_number_embedding(number):
    number_tensor = torch.tensor([number]).cuda()
    if number > 0:
        checker = number_encoder(number_tensor)
        if checker.dim() == 2 and checker.size(0) == 1:
            checker = checker.squeeze(0)
    else:
        checker = torch.zeros(args.hidden_size * 2)
    return checker


def format_string(s):
    return s.replace("b", "").replace("t", "").replace("\t", "")


def read_original_files(datapath: str):
    content = get_jsonl_content(datapath)
    result = []
    count = 0
    for item in content:
        print(f"{count} out of {len(content)}")
        count += 1
        law_label = article2id[str(item["meta"]["relevant_articles"][0])]
        accu_label = charge2id[str(item["meta"]["accusation"][0]).replace("[", "").replace("]", "")]
        amount = item["meta"]["crime_amount"]
        raw_fact = format_string(item["fact"])
        fact_cut = cutter.cut(raw_fact.strip(), text=True)

        # prison term label:
        if item["meta"]["term_of_imprisonment"]["death_penalty"] == True or item["meta"]["term_of_imprisonment"]["life_imprisonment"] == True:
            term_label = 0
        else:
            term_label = time2id[str(item["meta"]["term_of_imprisonment"]["imprisonment"])]
        result.append({"fact_cut": fact_cut, "article": law_label, "charge": accu_label, "term": term_label, "amount": amount})
    return result


def transform(word: str):
    if not (word in word2id.keys()):
        return word2id["UNK"]
    else:
        return word2id[word]


def sentence2index(sentence: str, mode):
    if mode == "fact":
        sent_len_max = args.fact_sentence_length
    elif mode == "charge":
        sent_len_max = args.charge_sentence_length
    elif mode == "article":
        sent_len_max = args.article_sentence_length
    else:
        sent_len_max = args.fact_sentence_length
    sent_tensor = torch.LongTensor(sent_len_max).zero_()
    word_list = sentence.split(" ")

    for w_id, word in enumerate(word_list):
        if w_id >= sent_len_max:break
        if word not in all_punc:
            sent_tensor[w_id] = transform(word)
    return sent_tensor


def transform_cail_files(mode: str):
    original_content = read_original_files("./data/original_data/" + mode + ".jsonl")
    # print(original_content[0])
    result = []
    for item in original_content:
        fact_sentence = sentence2index(item["fact_cut"], "fact")
        number_embedding = get_number_embedding(item["amount"])
        result.append({"fact": fact_sentence, "article_label": item["article"], "charge_label": item["charge"], "term_label": item["term"], "crime_amount": number_embedding})
    pk.dump(result, open("./data/tokenized_data/" + mode + ".pkl", "wb"))
    return


def encode_charge_and_article_definitions():
    charge_tong = get_json_content("./data/additional_data/charge_tong.json")
    charge_tong2id = {}
    id2charge_tong = {}
    for id, c in enumerate(charge_tong):
        charge_tong2id[c] = str(id)
        id2charge_tong[str(id)] = c
    
    article_tong = get_json_content("./data/additional_data/art_tong.json")
    article_tong2id = {}
    id2article_tong = {}
    for id, a in enumerate(article_tong):
        article_tong2id[a] = str(id)
        id2article_tong[str(id)] = a

    charge_details = get_json_content("./data/additional_data/charge_details.json")
    charge_definitions = []
    charge_definitions_dict = {}
    for i in charge_tong:
        current_definition = charge_details[i]["定义"].replace(" ", "")
        current_definition = format_string(current_definition)
        current_definition_cut = cutter.cut(current_definition.strip(), text=True)
        current_definition_embedded = sentence2index(current_definition_cut, "charge")
        charge_definitions.append(current_definition_embedded)
        charge_definitions_dict.update({charge_tong2id[i]: current_definition_embedded})
    # print(charge_definitions)
    
    article_details = get_json_content("./data/additional_data/law.json")
    law_definitions = []
    law_definitions_dict = {}
    for i in article_tong:
        current_definition = article_details[str(i)].replace(" ", "")
        current_definition = format_string(current_definition)
        current_definition_cut = cutter.cut(current_definition.strip(), text=True)
        current_definition_embedded = sentence2index(current_definition_cut, "article")
        law_definitions.append(current_definition_embedded)
        law_definitions_dict.update({article_tong2id[i]: current_definition_embedded})

    charge_definition_tensors = torch.stack(charge_definitions)
    law_definition_tensors = torch.stack(law_definitions)

    print(f"charge_definitions_shape: {charge_definition_tensors.shape}")
    print(f"law_definitions_shape: {law_definition_tensors.shape}")
    
    pk.dump(charge_definition_tensors, open("./data/tokenized_data/charge_definitions.pkl", "wb"))
    pk.dump(law_definition_tensors, open("./data/tokenized_data/law_definitions.pkl", "wb"))
    pk.dump(charge_definitions_dict, open("./data/tokenized_data/charge_definition_dict.pkl", "wb"))
    pk.dump(law_definitions_dict, open("./data/tokenized_data/law_definition_dict.pkl", "wb"))
    set_json_file(charge_tong2id, "./data/additional_data/charge_tong2id.json")
    set_json_file(id2charge_tong, "./data/additional_data/id2charge_tong.json")
    set_json_file(article_tong2id, "./data/additional_data/article_tong2id.json")
    set_json_file(id2article_tong, "./data/additional_data/id2article_tong.json")




if __name__ == "__main__":
    # modes = [
    #     "small_train", "small_valid", "small_test", "big_train", "big_valid", "big_test", 
    # ]
    # for mode in modes:
    #     transform_cail_files(mode)
    encode_charge_and_article_definitions()