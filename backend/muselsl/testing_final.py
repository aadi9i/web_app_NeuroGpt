import os
import numpy as np
import random
import torch
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from model import BrainTranslatorPreEncoder, BrainTranslator
import new_neuro

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def main_script(sentence):
    # sentence = 'Martyr gets royally screwed and comes back for more.'
    time_chunk_split_size = 3 # seconds for each word
    BUFFER_LENGTH = 5

    no_of_words = len(sentence.split(" "))
    print(f"EEG will be recorded for {no_of_words*time_chunk_split_size} seconds")
    new_neuro.extract_data_to_csv(no_of_words*time_chunk_split_size)

    df = pd.read_csv("my_data.csv")
    eeg_data = []
    for df_chunk in df.groupby(df.index // (time_chunk_split_size * BUFFER_LENGTH)):
            column_averages = list(df_chunk[1].mean())
            eeg_data.append(column_averages)
    while len(eeg_data) < 56:
            eeg_data.append([0]*32)
    EEG = torch.tensor([eeg_data])

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenized_input = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_input)
    if len(token_ids) < 56:
            token_ids = [0]+token_ids
    if len(token_ids) < 56:
            token_ids = token_ids+[2]  
    while len(token_ids) < 56:
            token_ids = token_ids+[0]  
    label = torch.tensor(token_ids).reshape(1, 56)

    input_masks = [True]*no_of_words + [False]*(56-no_of_words)
    input_masks_batch = torch.tensor(input_masks).reshape(1, 56)

    invert_masks = [False]*no_of_words + [True]*(56-no_of_words)
    input_masks_invert = torch.tensor(invert_masks).reshape(1, 56)

    cfg = {
            'seed': 312,
            'subject_choice': 'ALL',
            'eeg_type_choice': 'GD',
            'bands_choice': 'ALL',
            'dataset_setting': 'unique_sent',
            'batch_size': 1,
            'shuffle': False,
            'input_dim': 32,
            'num_layers': 1,  # 6
            'nhead': 1,  # 8
            'dim_pre_encoder': 2048,
            'dim_s2s': 1024,
            'dropout': 0,
            'T': 5e-6,
            'lr_pre': 1e-6,
            'epochs_pre': 5,
            'lr': 1e-6,
            'epochs': 5,
            'wandb': True
            }
    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load('temp_decoding_4_channels.pt',map_location=torch.device('cpu'))
    model = BrainTranslator(
            BrainTranslatorPreEncoder(
                input_dim=cfg['input_dim'],
                num_layers=cfg['num_layers'],
                nhead=cfg['nhead'],
                dim_pre_encoder=cfg['dim_pre_encoder'],
                dim_s2s=cfg['dim_s2s'],
                dropout=cfg['dropout']
                ).to(device),
            BartForConditionalGeneration.from_pretrained('facebook/bart-large'),
        ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    input_masks_batch = input_masks_batch.to(device)
    label = label.to(device)
    input_masks_invert_batch = input_masks_invert.to(device)
    label[label == tokenizer.pad_token_id] = -100
    output =  model(EEG, input_masks_batch , input_masks_invert , label)

    logits = output.logits  # bs*seq_len*voc_sz
    probs = logits.softmax(dim=-1)
    values, predictions = probs.topk(1)
    predictions = torch.squeeze(predictions, dim=-1)
    predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    print(f"Predicted String : {predicted_string}\n")

    return predicted_string
if __name__ == "__main__" :
       main_script()