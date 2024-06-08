import os
import numpy as np
import random
import torch
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from model import BrainTranslatorPreEncoder, BrainTranslator
import tensorflow as tf
from tensorflow.keras import layers, Model

# RAKESH'S MODEL
leny = 105
input_shape = (None, leny)
ind_1=1
ind_2=20
ind_3=47
ind_4=83

def convert_4_to_128(data):
    inp_arr = np.zeros((1, 8, 105))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            inp_arr[i][j][ind_1-1]=data[i][j][0]
            inp_arr[i][j][ind_2-1]=data[i][j][1]
            inp_arr[i][j][ind_3-1]=data[i][j][2]
            inp_arr[i][j][ind_4-1]=data[i][j][3]
    output = upsampler.predict(inp_arr)
    return np.concatenate(output[0]).reshape(-1)

# class CustomLayer(layers.Layer):
#     def __init__(self, **kwargs):
#         super(CustomLayer, self).__init__(**kwargs)

#     def call(self, inp):
#         mask = tf.concat([tf.zeros((tf.shape(inp)[0], ind_1-1)), tf.ones((tf.shape(inp)[0], 1)), tf.zeros((tf.shape(inp)[0], ind_2-ind_1-1)), tf.ones((tf.shape(inp)[0], 1)), tf.zeros((tf.shape(inp)[0], ind_3-ind_2-1)), tf.ones((tf.shape(inp)[0], 1)), tf.zeros((tf.shape(inp)[0], ind_4-ind_3-1)), tf.ones((tf.shape(inp)[0], 1)), tf.zeros((tf.shape(inp)[0], leny-ind_4))], axis=-1)
#         masked_input = layers.Multiply()([inp, mask])
#         return masked_input


def mask_function(inp):
    ind_1=1
    ind_2=20
    ind_3=47
    ind_4=83
    leny = 105
    mask = tf.concat([
        tf.zeros((tf.shape(inp)[0], ind_1-1)),
        tf.ones((tf.shape(inp)[0], 1)),
        tf.zeros((tf.shape(inp)[0], ind_2-ind_1-1)),
        tf.ones((tf.shape(inp)[0], 1)),
        tf.zeros((tf.shape(inp)[0], ind_3-ind_2-1)),
        tf.ones((tf.shape(inp)[0], 1)),
        tf.zeros((tf.shape(inp)[0], ind_4-ind_3-1)),
        tf.ones((tf.shape(inp)[0], 1)),
        tf.zeros((tf.shape(inp)[0], leny-ind_4))
    ], axis=-1)
    
    return inp * mask


def ups(inp):
    # x = CustomLayer()(inp)
    x = layers.Lambda(mask_function, output_shape=lambda x: x)(inp)
    
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    y = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(48, 3, activation="relu", padding="same")(x+y)
    
    x = layers.Conv1D(48, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    # Upsampling layers
    # x = layers.UpSampling1D(size=2)(x)
    # x = layers.Reshape((32, 16))(x)
    # Additional convolutional layers for depth
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Reshape((8, 64))(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    y = layers.Conv1D(leny, 3, activation="sigmoid", padding="same")(x)
    z = layers.Conv1D(leny, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(leny, 3, activation="relu", padding="same")(y+z)
    x = layers.Conv1D(leny, 3, activation="relu", padding="same")(x)
    
    res = layers.Conv1D(leny, 3, activation="sigmoid", padding="same")(x)
    
    return res

inp = tf.keras.Input(shape=input_shape)
resu = ups(inp)
upsampler = Model(inputs=inp, outputs=resu)

upsampler.load_weights("autoencoder_4_to_128_channels_best_1.h5")


# CODE ENDS

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
def pre(sentence):

    time_chunk_split_size = 3 # seconds for each word
    BUFFER_LENGTH = 5

    no_of_words = len(sentence.split(" "))

    df = pd.read_csv("my_data.csv")
    eeg_data = []
    for df_chunk in df.groupby(df.index // (time_chunk_split_size * BUFFER_LENGTH)):
        column_averages = list(df_chunk[1].mean())
        matrix = np.array([[column_averages[i:i+4] for i in range(0, len(column_averages), 4)]])
        upsampled = convert_4_to_128(matrix)
        eeg_data.append(upsampled)
        print(len(upsampled))
    while len(eeg_data) < 56:
        eeg_data.append([0]*840)
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
        'input_dim': 840,
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

    state_dict = torch.load('temp_decoding.pt',map_location=torch.device('cpu'))
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
    return(predicted_string)
if __name__ == "__main__" :
    run()