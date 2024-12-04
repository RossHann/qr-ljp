from torch.utils.data import Dataset, DataLoader, RandomSampler
import logging
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import json
import random
import numpy as np
from parameters_lawformer import parse
from utils import EarlyStopping
from model import TCALJP_Lawformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from file_operations import get_jsonl_content
import pickle as pk



class CAILDataset_Original(Dataset):
    def __init__(self, file_path, args):
        content = pk.load(open(file_path, "rb"))
        if args.small_sample:
            self.content = content[:args.small_sample_size]
        else:
            self.content = content
        
    def __getitem__(self, index):
        current_item = self.content[index]
        fact = current_item["fact"]
        crime_amount = current_item["crime_amount"].cuda()
        if crime_amount.dim() == 2 and crime_amount.size(0) == 1:
            crime_amount = crime_amount.squeeze(0)
        return fact, current_item["charge_label"], current_item["article_label"], current_item["term_label"], crime_amount
    
    def __len__(self):
        return len(self.content)
    

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger(args):
    logger = logging.getLogger("current_logger")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{formatted_time}.log")
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def get_metrics(targets, predictions, logger):
    acc = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=1)
    macro_recall = recall_score(targets, predictions, average="macro", zero_division=1)
    macro_precision = precision_score(targets, predictions, average="macro", zero_division=1)

    logger.info(f"Accuracy: {acc:.5f}, Macro Precision: {macro_precision:.5f}, Macro Recall: {macro_recall:.5f}, Macro F1: {macro_f1:.5f}")
    # return acc, macro_precision, macro_recall, macro_f1
    return macro_f1


def main(args):
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices
    logger = set_logger(args)
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {n_gpu}")
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = TCALJP_Lawformer(args)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Parameter {name} requires grandients.")
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model)

    train_data = CAILDataset_Original(args.train_datapath, args)
    valid_data = CAILDataset_Original(args.valid_datapath, args)
    test_data = CAILDataset_Original(args.test_datapath, args)

    train_sampler = RandomSampler(train_data)
    valid_sampler = RandomSampler(valid_data)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, drop_last=False)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, sampler=valid_sampler, num_workers=0, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    l_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=l_rate)
    max_epochs = args.max_epochs
    early_stopping = EarlyStopping(args.model_savepath, patience=args.patience)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        logger.info("Training Epoch: {}/{}.".format(epoch + 1, int(max_epochs)))
        for step, batch in enumerate(tqdm(train_dataloader)):
            nb_tr_steps += 1
            # print(batch)
            model.train()
            optimizer.zero_grad()
            fact = batch[0]
            charge_label = batch[1].cuda()
            article_label = batch[2].cuda()
            term_label = batch[3].cuda()
            crime_amount = batch[4].cuda()
            # print(f"charge_label: shape-{charge_label.shape}")
            # print(f"article_label: shape-{article_label.shape}")
            # print(f"term_label: shape-{term_label.shape}")
            # print(f"crime_amount: shape-{crime_amount.shape}")
            charge_out_logits, article_out_logits, term_out_logits = model(fact, crime_amount)
            loss_charge = criterion(charge_out_logits, charge_label)
            loss_article = criterion(article_out_logits, article_label)
            loss_term = criterion(term_out_logits, term_label)
            loss = (loss_charge + loss_article + loss_term) / 3
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
        logger.info(f"Epoch: {epoch + 1}, train_loss: {tr_loss / len(train_dataloader)}")
        charge_pred, article_pred, term_pred, charge_true, article_true, term_true = [], [], [], [], [], []
        model.eval()
        val_loss = 0
        for step, batch in enumerate(tqdm(valid_dataloader)):
            model.eval()
            fact = batch[0]
            charge_label = batch[1].cuda()
            article_label = batch[2].cuda()
            term_label = batch[3].cuda()
            crime_amount = batch[4].cuda()
            with torch.no_grad():
                charge_out_logits, article_out_logits, term_out_logits = model(fact, crime_amount)
            loss_charge = criterion(charge_out_logits, charge_label)
            loss_article = criterion(article_out_logits, article_label)
            loss_term = criterion(term_out_logits, term_label)
            loss = (loss_charge + loss_article + loss_term) / 3
            val_loss += loss.item()
            charge_pred.extend(torch.argmax(charge_out_logits, dim=-1).cpu().numpy())
            article_pred.extend(torch.argmax(article_out_logits, dim=-1).cpu().numpy())
            term_pred.extend(torch.argmax(term_out_logits, dim=-1).cpu().numpy())
            charge_true.extend(charge_label.cpu().numpy())
            article_true.extend(article_label.cpu().numpy())
            term_true.extend(term_label.cpu().numpy())
        logger.info(f"Epoch: {epoch + 1}, valid_loss: {val_loss / len(valid_dataloader)}")
        logger.info(f"Charge Prediction")
        charge_macro_f1 = get_metrics(charge_pred, charge_true, logger)
        logger.info(f"Article Prediction")
        article_macro_f1 = get_metrics(article_pred, article_true, logger)
        logger.info(f"Term Prediction")
        term_macro_f1 = get_metrics(term_pred, term_true, logger)
        if early_stopping is not None:
            # stop = early_stopping(term_macro_f1, model)
            stop = early_stopping(val_loss, model)
            if stop:
                logger.info(f"Early Stopping...")
                break
    logger.info("Starting inference on the test dataset.")
    charge_pred_inf, article_pred_inf, term_pred_inf, charge_true_inf, article_true_inf, term_true_inf = [], [], [], [], [], []
    model.eval()
    test_loss = 0
    for step, batch in enumerate(tqdm(test_dataloader)):
        model.eval()
        fact = batch[0]
        charge_label = batch[1].cuda()
        article_label = batch[2].cuda()
        term_label = batch[3].cuda()
        crime_amount = batch[4].cuda()
        with torch.no_grad():
            charge_out_logits, article_out_logits, term_out_logits = model(fact, crime_amount)
        loss_charge = criterion(charge_out_logits, charge_label)
        loss_article = criterion(article_out_logits, article_label)
        loss_term = criterion(term_out_logits, term_label)
        loss = (loss_charge + loss_article + loss_term) / 3
        test_loss += loss.item()
        charge_pred_inf.extend(torch.argmax(charge_out_logits, dim=-1).cpu().numpy())
        article_pred_inf.extend(torch.argmax(article_out_logits, dim=-1).cpu().numpy())
        term_pred_inf.extend(torch.argmax(term_out_logits, dim=-1).cpu().numpy())
        charge_true_inf.extend(charge_label.cpu().numpy())
        article_true_inf.extend(article_label.cpu().numpy())
        term_true_inf.extend(term_label.cpu().numpy())
    logger.info(f"Testing, test_loss: {test_loss / len(test_dataloader)}")
    logger.info(f"Charge Prediction")
    charge_macro_f1 = get_metrics(charge_pred_inf, charge_true_inf, logger)
    logger.info(f"Article Prediction")
    article_macro_f1 = get_metrics(article_pred_inf, article_true_inf, logger)
    logger.info(f"Term Prediction")
    term_macro_f1 = get_metrics(term_pred_inf, term_true_inf, logger)
    return


if __name__ == "__main__":
    args = parse()
    main(args)
