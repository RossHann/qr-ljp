import thulac
import torch
import os
import torch.nn as nn
import pickle as pk
from parameters_lawformer import parse
from file_operations import get_json_content, get_jsonl_content, set_json_file
from string import punctuation
from pretrained_model.models import NumberEncoder
from transformers import AutoModel, AutoTokenizer

args = parse()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
# print("youre so stupid")

number_encoder = NumberEncoder(args)
if torch.cuda.is_available():
    number_encoder = nn.DataParallel(number_encoder)
    number_encoder.cuda()
number_encoder.load_state_dict(torch.load("./pretrained_model/number_encoder"))

tokenizer = AutoTokenizer.from_pretrained("pretrained_model/lawformer")

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
        with torch.no_grad():
            checker = number_encoder(number_tensor)
        # torch.no_grad()
        if checker.dim() == 2 and checker.size(0) == 1:
            checker = checker.squeeze(0)
    else:
        checker = torch.zeros(args.hidden_size * 2)
    # answer = checker
    # del checker
    del number_tensor
    torch.cuda.empty_cache()
    return checker

def format_string(s):
    return s.replace("b", "").replace("t", "").replace("\t", "")


def read_original_files(datapath: str):
    content = get_jsonl_content(datapath)
    result = []
    count = 0
    for item in content:
        print(f"{count + 1} out of {len(content)}")
        count += 1
        law_label = article2id[str(item["meta"]["relevant_articles"][0])]
        accu_label = charge2id[str(item["meta"]["accusation"][0]).replace("[", "").replace("]", "")]
        amount = item["meta"]["crime_amount"]
        raw_fact = format_string(item["fact"])

        if item["meta"]["term_of_imprisonment"]["death_penalty"] == True or item["meta"]["term_of_imprisonment"]["life_imprisonment"] == True:
            term_label = 0
        else:
            term_label = time2id[str(item["meta"]["term_of_imprisonment"]["imprisonment"])]
        result.append({"fact": raw_fact, "article": law_label, "charge": accu_label, "term": term_label, "amount": amount})
        del raw_fact
        torch.cuda.empty_cache()
    return result


def transform_cail_files(mode: str):
    original_content = read_original_files("./data/original_data/" + mode + ".jsonl")
    # print(original_content[0])
    result = []
    count = 0
    for item in original_content:
        print(f"{count + 1} out of {len(original_content)}")
        count += 1
        number_embedding = get_number_embedding(item["amount"])
        # get_number_embedding(item["amount"])
        # torch.cuda.empty_cache()
        fact_ids = tokenizer(item["fact"], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        # print(len(fact_ids["input_ids"]))
        # number_embedding_cpu = number_embedding.cpu()
        # del number_embedding
        result.append({"fact": fact_ids, "article_label": item["article"], "charge_label": item["charge"], "term_label": item["term"], "crime_amount": number_embedding})
        
        # result.append({"fact": fact_ids, "article_label": item["article"], "charge_label": item["charge"], "term_label": item["term"]})
        del fact_ids
        del number_embedding
        torch.cuda.empty_cache()
    pk.dump(result, open("./data/tokenized_data_lawformer/" + mode + ".pkl", "wb"))


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
        current_definition_embedded = tokenizer(current_definition, max_length=args.charge_sentence_length, truncation=True, padding="max_length", return_tensors="pt")
        charge_definitions.append(current_definition_embedded)
        charge_definitions_dict.update({charge_tong2id[i]: current_definition_embedded})

    article_details = get_json_content("./data/additional_data/law.json")
    law_definitions = []
    law_definitions_dict = {}
    for i in article_tong:
        current_definition = article_details[str(i)].replace(" ", "")
        current_definition = format_string(current_definition)
        current_definition_embedded = tokenizer(current_definition, max_length=args.article_sentence_length, truncation=True, padding="max_length", return_tensors="pt")
        law_definitions.append(current_definition_embedded)
        law_definitions_dict.update({article_tong2id[i]: current_definition_embedded})
    print(charge_definitions[0])

    tmp_input_ids_charge = []
    tmp_attention_masks_charge = []
    for item in charge_definitions:
        tmp_input_ids_charge.append(item["input_ids"])
        tmp_attention_masks_charge.append(item["attention_mask"])
    input_ids_tensor_charge = torch.stack(tmp_input_ids_charge)
    attention_masks_tensor_charge = torch.stack(tmp_attention_masks_charge)
    charge_definitions_tensor = {"input_ids": input_ids_tensor_charge, "attention_mask": attention_masks_tensor_charge}

    tmp_input_ids_article = []
    tmp_attention_masks_article = []
    for item in law_definitions:
        tmp_input_ids_article.append(item["input_ids"])
        tmp_attention_masks_article.append(item["attention_mask"])
    input_ids_tensor_article = torch.stack(tmp_input_ids_article)
    attention_masks_tensor_article = torch.stack(tmp_attention_masks_article)
    article_definition_tensor = {"input_ids": input_ids_tensor_article, "attention_mask": attention_masks_tensor_article}


    pk.dump(charge_definitions_dict, open("./data/tokenized_data_lawformer/charge_definition_dict.pkl", "wb"))
    pk.dump(law_definitions_dict, open("./data/tokenized_data_lawformer/law_definition_dict.pkl", "wb"))
    set_json_file(charge_tong2id, "./data/additional_data/charge_tong2id_lawformer.json")
    set_json_file(id2charge_tong, "./data/additional_data/id2charge_tong_lawformer.json")
    set_json_file(article_tong2id, "./data/additional_data/article_tong2id_lawformer.json")
    set_json_file(id2article_tong, "./data/additional_data/id2article_tong_lawformer.json")
    pk.dump(charge_definitions_tensor, open("./data/tokenized_data_lawformer/charge_definitions_lawformer.pkl", "wb"))
    pk.dump(article_definition_tensor, open("./data/tokenized_data_lawformer/law_definitions_lawformer.pkl", "wb"))
    pk.dump(charge_definitions, open("./data/tokenized_data_lawformer/charge_definition_lawformer_per_item.pkl", "wb"))
    pk.dump(law_definitions, open("./data/tokenized_data_lawformer/law_definitions_lawformer_per_item.pkl", "wb"))

# def null():
#     print("youre stupid")


if __name__ == "__main__":
    modes = [
        "small_train", "small_valid", "small_test", "big_train", "big_valid", "big_test", 
    ]
    for mode in modes[4:]:
        transform_cail_files(mode)
    # encode_charge_and_article_definitions()
    # null()