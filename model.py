import torch 
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle as pk
import numpy as np
from transformers import AutoModel, AutoTokenizer


class Numerical_Attention(nn.Module):
    def __init__(self):
        super(Numerical_Attention, self).__init__()

    def forward(self, query, context):
        query_expand = query.unsqueeze(1)
        # print(f"query_expand: shape-{query_expand.shape}")
        # if torch.isnan(query_expand).any() or torch.isinf(query_expand).any():print(f"query_expand has nan or inf within.")
        S = torch.bmm(query_expand, context.transpose(1, 2))
        if torch.isnan(S).any() or torch.isinf(S).any():
            # print(f"S: content-{S.cpu().numpy()}")
            S = torch.nan_to_num(S)
            # print(f"query: content-{query}")
            # ss = S
            # ss_list = ss.detach().cpu().numpy().tolist()
            # print(f"ss_list: content-{ss_list}")
        # print(f"S: shape-{S.shape}")
        attention = F.softmax(S, dim=-1)
        # if torch.isnan(attention).any():print(f"attention has nan within.")
        # print(f"attention: shape-{attention.shape}")
        # 改
        context_vec = torch.bmm(attention, context)
        # print(f"context_vec: shape-{context_vec.shape}")
        # context_vec = context_vec.repeat(1, context.size(1), 1)
        return context_vec


class Textual_Attention(nn.Module):
    def __init__(self):
        super(Textual_Attention, self).__init__()

    def forward(self, query, context):
        # print(f"query shape-{query.shape}")
        # print(f"context shape-{context.shape}")
        S = torch.bmm(context, query.transpose(1, 2))
        attention = F.softmax(torch.max(S, 2)[0], dim=-1)
        context_vec = torch.bmm(attention.unsqueeze(1), context)
        return context_vec


class TCALJP(nn.Module):
    def __init__(self, args):
        super(TCALJP, self).__init__()
        self.args = args

        self.embedding_weights = np.load(self.args.embedding_path)
        self.embedding_weights_tensor = torch.from_numpy(self.embedding_weights).float()
        self.embedding_layer = nn.Embedding.from_pretrained(self.embedding_weights_tensor)
        self.embedding_layer.weight.requires_grad = False

        self.charge_tong2id = json.load(open("./data/additional_data/charge_tong2id.json"))
        self.article_tong2id = json.load(open("./data/additional_data/article_tong2id.json"))
        self.id2charge_tong = json.load(open("./data/additional_data/id2charge_tong.json"))
        self.id2article_tong = json.load(open("./data/additional_data/id2article_tong.json"))
        self.charge_descriptions = pk.load(open("./data/tokenized_data/charge_definitions.pkl", "rb"))
        self.article_descriptions = pk.load(open("./data/tokenized_data/law_definitions.pkl", "rb"))
        self.charge_descriptions_dict = pk.load(open("./data/tokenized_data/charge_definition_dict.pkl", "rb"))
        self.article_descriptions_dict = pk.load(open("./data/tokenized_data/law_definition_dict.pkl", "rb"))
        self.charge_description_embedded = self.embedding_layer(self.charge_descriptions)
        self.article_description_embedded = self.embedding_layer(self.article_descriptions)

        self.shared_encoder = nn.GRU(input_size=self.args.embedding_dims, batch_first=True, hidden_size=self.args.hidden_size, bidirectional=True)
        
        self.textual_attention = Textual_Attention()
        self.numerical_attention = Numerical_Attention()

        self.charge_prediction = nn.Linear(self.args.hidden_size * 4, self.charge_description_embedded.size(0))
        self.article_prediction = nn.Linear(self.args.hidden_size * 6, self.article_description_embedded.size(0))
        self.term_prediction = nn.Linear(self.args.hidden_size * 4, 11)



    def forward(self, fact, crime_amount):
        # print(f"fact init: type-{type(fact)} shape-{fact.shape}")
        # print(f"crime amount init: type-{type(crime_amount)} shape-{crime_amount.shape}")
        self.shared_encoder.flatten_parameters()
        self.charge_description_embedded = self.charge_description_embedded.cuda()
        self.article_description_embedded = self.article_description_embedded.cuda()
        fact_embedded = self.embedding_layer(fact)
        fact_hidden, _ = self.shared_encoder(fact_embedded)
        # print(f"charge_descriptions: {self.charge_descriptions.shape}")
        # print(f"article_descriptions: {self.article_descriptions.shape}")
        charge_description_hidden, _ = self.shared_encoder(self.charge_description_embedded)
        article_description_hidden, _ = self.shared_encoder(self.article_description_embedded)
        # print(f"f_hidden: shape-{f_hidden.shape}")
        # print(f"charge_description_hidden: shape-{charge_description_hidden.shape}")
        # print(f"article_description_hidden: shape-{article_description_hidden.shape}")

        charge_description_hidden_sentence_mean = charge_description_hidden.mean(1)
        charge_description_hidden_sentence_mean_batched = charge_description_hidden_sentence_mean.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        fact_charge_arranged = self.textual_attention(charge_description_hidden_sentence_mean_batched, fact_hidden)
        fact_charge_arranged = fact_charge_arranged.repeat(1, fact_hidden.size(1), 1)
        # print(f"fact_charge_arranged shape-{fact_charge_arranged.shape}")

        charge_input = torch.cat([fact_hidden, fact_charge_arranged], dim=-1)
        # print(f"fact_input shape-{fact_input.shape}")
        charge_input_mean = charge_input.mean(1)
        charge_out = self.charge_prediction(charge_input_mean)
        charge_out_np = charge_out.argmax(dim=1).cpu().numpy()
        charge_predicted_list = [self.charge_descriptions_dict[str(i)] for i in charge_out_np]
        charge_predicted_embedding_stack = torch.stack(charge_predicted_list).cuda()
        charge_predicted_tensor_stack = self.embedding_layer(charge_predicted_embedding_stack)
        # print(f"charge_prediction_tensor_stack: -{charge_predicted_tensor_stack.shape}")
        charge_predicted_tensor_stack_hidden, _ = self.shared_encoder(charge_predicted_tensor_stack)
        # print(f"charge_predicted_tensor_stack_hidden: -{charge_predicted_tensor_stack_hidden.shape}")
        diag_charge = torch.ones(charge_predicted_tensor_stack_hidden.size(1)).cuda()
        # diag_charge = torch.ones(3)
        diag_matrix_charge = torch.diag_embed(diag_charge)
        full_diag_matrix_charge = torch.zeros(self.args.hidden_size * 2, charge_predicted_tensor_stack_hidden.size(1)).cuda()
        # full_diag_matrix_charge = torch.zeros(6, 3)
        full_diag_matrix_charge[:charge_predicted_tensor_stack_hidden.size(1), :] = diag_matrix_charge
        # full_diag_matrix_charge[:3, :] = diag_matrix_charge
        # print(f"full_diag_matrix_charge: shape-{full_diag_matrix_charge.shape}")
        full_diag_matrix_charge_batched = full_diag_matrix_charge.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        charge_predicted_tensor_stack_hidden_reshaped = torch.bmm(full_diag_matrix_charge_batched, charge_predicted_tensor_stack_hidden)
        # print(f"charge_predicted_tensor_stack_hidden_reshaped: shape-{charge_predicted_tensor_stack_hidden_reshaped.shape}")

        article_description_hidden_sentence_mean = article_description_hidden.mean(1)
        article_description_hidden_sentence_mean_batched = article_description_hidden_sentence_mean.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        fact_article_arranged = self.textual_attention(article_description_hidden_sentence_mean_batched, fact_hidden)
        fact_article_arranged = fact_article_arranged.repeat(1, fact_hidden.size(1), 1)

        article_input = torch.cat([fact_hidden, fact_article_arranged, charge_predicted_tensor_stack_hidden_reshaped], dim=-1)
        article_input_mean = article_input.mean(1)
        article_out = self.article_prediction(article_input_mean)
        article_out_np = article_out.argmax(dim=1).cpu().numpy()
        article_predicted_list = [self.article_descriptions_dict[str(i)] for i in article_out_np]
        article_predicted_embedding_stack = torch.stack(article_predicted_list).cuda()
        article_predicted_tensor_stack = self.embedding_layer(article_predicted_embedding_stack)
        # print(f"article_predicted_tensor_stack: shape-{article_predicted_tensor_stack.shape}")
        article_predicted_tensor_stack_hidden, _ = self.shared_encoder(article_predicted_tensor_stack)
        # print(f"article_predicted_tensor_stack_hidden: shape-{article_predicted_tensor_stack_hidden.shape}")
        # print(f"crime_amount: shape-{crime_amount.shape}")
        # if torch.isnan(article_predicted_tensor_stack_hidden).any() or torch.isinf(article_predicted_tensor_stack_hidden).any():print(f"article_predicted_tensor_stack_hidden has nan or inf within.")
        article_predicted_tensor_stack_hidden_rearranged = self.numerical_attention(crime_amount, article_predicted_tensor_stack_hidden)
        # if torch.isnan(article_predicted_tensor_stack_hidden_rearranged).any():print(f"article_predicted_tensor_stack_hidden_rearranged has nan within.")
        # if torch.isinf(article_predicted_tensor_stack_hidden_rearranged).any():print(f"article_predicted_tensor_stack_hidden_rearranged has inf within.")
        article_predicted_tensor_stack_hidden_rearranged = article_predicted_tensor_stack_hidden_rearranged.repeat(1, fact_hidden.size(1), 1)
        # print(f"article_predicted_tensor_stack_hidden_rearranged: shape-{article_predicted_tensor_stack_hidden_rearranged.shape}")
        diag_article = torch.ones(article_predicted_tensor_stack_hidden_rearranged.size(1)).cuda()
        diag_matrix_article = torch.diag_embed(diag_article)
        full_diag_matrix_article = torch.zeros(self.args.hidden_size * 2, article_predicted_tensor_stack_hidden_rearranged.size(1)).cuda()
        full_diag_matrix_article[:article_predicted_tensor_stack_hidden_rearranged.size(1), :] = diag_matrix_article
        full_diag_matrix_article_batched = full_diag_matrix_article.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        # if torch.isnan(article_predicted_tensor_stack_hidden_rearranged).any() or torch.isinf(article_predicted_tensor_stack_hidden_rearranged).any():print(f"article_predicted_tensor_stack_hidden_rearranged has nan or inf within.")

        article_predicted_tensor_stack_hidden_reshaped = torch.bmm(full_diag_matrix_article_batched, article_predicted_tensor_stack_hidden_rearranged)
        # print(f"article_predicted_tensor_stack_hidden_reshaped: shape-{article_predicted_tensor_stack_hidden_reshaped.shape}")

        # if torch.isnan(article_predicted_tensor_stack_hidden_reshaped).any() or torch.isinf(article_predicted_tensor_stack_hidden_reshaped).any():print(f"article_predicted_tensor_stack_hidden_reshaped has nan or inf within.")
        
        term_input = torch.cat([fact_hidden, article_predicted_tensor_stack_hidden_reshaped], dim=-1)
        # if torch.isnan(term_input).any() or torch.isinf(term_input).any():print(f"term_input has nan or inf within.")
        term_input_mean = term_input.mean(1)
        # if torch.isnan(term_input_mean).any() or torch.isinf(term_input_mean).any():print(f"term_input_mean has nan or inf within.")
        term_out = self.term_prediction(term_input_mean)
        # if torch.isnan(term_out).any() or torch.isinf(term_out).any():print(f"term_out has nan or inf within.")
        # print(f"charge_out: shape-{charge_out.shape}, requires_grad-{charge_out.requires_grad}")
        # print(f"article_out: shape-{article_out.shape}")
        # print(f"term_out: shape-{term_out.shape}")
        # charge_out_argmax = charge_out.argmax(dim=1).float()
        # article_out_argmax = article_out.argmax(dim=1).float()
        # term_out_argmax = term_out.argmax(dim=1).float()
        # print(f"charge_out_argmax: shape-{charge_out_argmax.shape}, requires_grad-{charge_out_argmax.requires_grad}")
        # return charge_out_argmax, article_out_argmax, term_out_argmax
        return charge_out, article_out, term_out


class TCALJP_Lawformer(nn.Module):
    def __init__(self, args):
        super(TCALJP_Lawformer, self).__init__()
        self.args = args
        # self.tmp_linear = nn.Linear(256, 12)
        self.shared_encoder = AutoModel.from_pretrained("pretrained_model/lawformer")
        for name, param in self.shared_encoder.named_parameters():
            if not "layer.11" in name:
                param.requires_grad = False
        self.linear_projector = nn.Linear(768, self.args.hidden_size * 2)

        self.charge_description = pk.load(open("./data/tokenized_data_lawformer/charge_definition_lawformer_per_item.pkl", "rb"))
        self.article_description = pk.load(open("./data/tokenized_data_lawformer/law_definitions_lawformer_per_item.pkl", "rb"))
        self.charge_descriptions_dict = pk.load(open("./data/tokenized_data_lawformer/charge_definition_dict.pkl", "rb"))
        self.article_descriptions_dict = pk.load(open("./data/tokenized_data_lawformer/law_definition_dict.pkl", "rb"))
        
        
        self.textual_attention = Textual_Attention()
        self.numerical_attention = Numerical_Attention()

        self.charge_prediction = nn.Linear(self.args.hidden_size * 4, len(self.charge_description))
        self.article_prediction = nn.Linear(self.args.hidden_size * 6, len(self.charge_description))
        self.term_prediction = nn.Linear(self.args.hidden_size * 4, 11)
    
    def forward(self, fact, crime_amount):
        charge_tensor_list = []
        for item in self.charge_description:
            current_input_ids = item["input_ids"].cuda()
            current_input_ids = current_input_ids.squeeze(1)
            current_attention_mask = item["attention_mask"].cuda()
            current_attention_mask = current_attention_mask.squeeze(1)
            current_charge_on_gpu = {"input_ids": current_input_ids, "attention_mask": current_attention_mask}
            # print(f"current_input_ids: shape-{current_input_ids.shape}")
            encoded_current_charge = self.shared_encoder(**current_charge_on_gpu).last_hidden_state
            encoded_current_charge_arranged = self.linear_projector(encoded_current_charge)
            charge_tensor_list.append(encoded_current_charge_arranged)
        charge_description_hidden = torch.stack(charge_tensor_list)
        # print(f"charge_description_hidden: shape-{charge_description_hidden.shape}")
        article_tensor_list = []
        for item in self.article_description:
            current_input_ids = item["input_ids"].cuda()
            current_input_ids = current_input_ids.squeeze(1)
            current_attention_mask = item["attention_mask"].cuda()
            current_attention_mask = current_attention_mask.squeeze(1)
            current_article_on_gpu = {"input_ids": current_input_ids, "attention_mask": current_attention_mask}
            # print(f"current_input_ids: shape-{current_input_ids.shape}")
            encoded_current_article = self.shared_encoder(**current_article_on_gpu).last_hidden_state
            encoded_current_article_arranged = self.linear_projector(encoded_current_article)
            article_tensor_list.append(encoded_current_article_arranged)
        article_description_hidden = torch.stack(article_tensor_list)
        # print(f"article_description_hidden: shape-{article_description_hidden.shape}")
        # print(f"fact: type-{type(fact)}, content-{fact}")
        fact_input_ids = fact["input_ids"].cuda()
        fact_input_ids = fact_input_ids.squeeze(1)
        # # print(f"fact_input_ids: shape-{fact_input_ids.shape}, content-{fact_input_ids}")
        fact_attention_mask = fact["attention_mask"].cuda()
        fact_attention_mask = fact_attention_mask.squeeze(1)
        # # print(f"fact_attention_mask: shape-{fact_attention_mask.shape}")
        fact_on_gpu = {"input_ids": fact_input_ids, "attention_mask": fact_attention_mask}
        encoded_fact = self.shared_encoder(**fact_on_gpu).last_hidden_state
        fact_hidden = self.linear_projector(encoded_fact)
        # print(f"encoded_fact_arranged: shape-{encoded_fact_arranged.shape}")
        
        # print(f"charge_description_hidden: shape-{charge_description_hidden.shape}")
        charge_description_hidden_sentence_mean = charge_description_hidden.mean(1)
        charge_description_hidden_sentence_mean = charge_description_hidden_sentence_mean.mean(1)
        # print(f"charged_description_hidden_sentence_mean: shape-{charge_description_hidden_sentence_mean.shape}")
        charge_description_hidden_sentence_mean_batched = charge_description_hidden_sentence_mean.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        fact_charge_arranged = self.textual_attention(charge_description_hidden_sentence_mean_batched, fact_hidden)
        fact_charge_arranged = fact_charge_arranged.repeat(1, fact_hidden.size(1), 1)
        
        charge_input = torch.cat([fact_hidden, fact_charge_arranged], dim=-1)
        charge_input_mean = charge_input.mean(1)
        charge_out = self.charge_prediction(charge_input_mean)
        charge_out_np = charge_out.argmax(dim=1).cpu().numpy()
        charge_predicted_list = [self.charge_descriptions_dict[str(i)] for i in charge_out_np]
        # print(f"charge_prediction_list: length-{len(charge_predicted_list)}")
        predicted_charge_input_ids = charge_predicted_list[0]["input_ids"]
        predicted_charge_input_ids = predicted_charge_input_ids.cuda()
        predicted_charge_attention_mask = charge_predicted_list[0]["attention_mask"]
        predicted_charge_attention_mask = predicted_charge_attention_mask.cuda()
        predicted_charge_on_gpu = {"input_ids": predicted_charge_input_ids, "attention_mask": predicted_charge_attention_mask}
        encoded_predicted_charge = self.shared_encoder(**predicted_charge_on_gpu).last_hidden_state
        encoded_predicted_charge_arranged = self.linear_projector(encoded_predicted_charge)
        # print(f"encoded_predicted_charge: shape-{encoded_predicted_charge.shape}")       

        diag_charge = torch.ones(encoded_predicted_charge_arranged.size(1)).cuda()
        diag_matrix_charge = torch.diag_embed(diag_charge)
        full_diag_matrix_charge = torch.zeros(self.args.hidden_size * 2, encoded_predicted_charge_arranged.size(1)).cuda()
        full_diag_matrix_charge[:encoded_predicted_charge_arranged.size(1), :] = diag_matrix_charge
        full_diag_matrix_charge_batched = full_diag_matrix_charge.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        charge_predicted_tensor_stack_hidden_reshaped = torch.bmm(full_diag_matrix_charge_batched, encoded_predicted_charge_arranged)

        # print(f"charge_predicted_tensor_stack_hidden_reshaped: shape-{charge_predicted_tensor_stack_hidden_reshaped.shape}")

        article_description_hidden_mean = article_description_hidden.mean(1)
        article_description_hidden_mean = article_description_hidden_mean.mean(1)
        article_description_hidden_sentence_mean_batched = article_description_hidden_mean.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        fact_article_arranged = self.textual_attention(article_description_hidden_sentence_mean_batched, fact_hidden)
        fact_article_arranged = fact_article_arranged.repeat(1, fact_hidden.size(1), 1)
        article_input = torch.cat([fact_hidden, fact_article_arranged, charge_predicted_tensor_stack_hidden_reshaped], dim=-1)
        article_input_mean = article_input.mean(1)
        article_out = self.article_prediction(article_input_mean)
        article_out_np = article_out.argmax(dim=1).cpu().numpy()
        article_predicted_list = [self.article_descriptions_dict[str(i)] for i in article_out_np]
        predicted_article_input_ids = article_predicted_list[0]["input_ids"]
        predicted_article_input_ids = predicted_article_input_ids.cuda()
        predicted_article_attention_mask = article_predicted_list[0]["attention_mask"]
        predicted_article_attention_mask = predicted_article_attention_mask.cuda()
        predicted_article_on_gpu = {"input_ids": predicted_article_input_ids, "attention_mask": predicted_article_attention_mask}
        encoded_predicted_article = self.shared_encoder(**predicted_article_on_gpu).last_hidden_state
        encoded_predicted_article_arranged = self.linear_projector(encoded_predicted_article)

        diag_article = torch.ones(encoded_predicted_article_arranged.size(1)).cuda()
        diag_matrix_article = torch.diag_embed(diag_article)
        full_diag_matrix_article = torch.zeros(self.args.hidden_size * 2, encoded_predicted_article_arranged.size(1)).cuda()
        full_diag_matrix_article[:encoded_predicted_article_arranged.size(1), :] = diag_matrix_article
        full_diag_matrix_article_batched = full_diag_matrix_article.unsqueeze(0).repeat(fact_hidden.size(0), 1, 1)
        article_predicted_tensor_stack_hidden_reshaped = torch.bmm(full_diag_matrix_article_batched, encoded_predicted_article_arranged)

        term_input = torch.cat([fact_hidden, article_predicted_tensor_stack_hidden_reshaped], dim=-1)
        term_input_mean = term_input.mean(1)
        term_out = self.term_prediction(term_input_mean)
        return charge_out, article_out, term_out