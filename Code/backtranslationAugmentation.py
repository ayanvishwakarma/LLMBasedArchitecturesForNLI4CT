import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import json
import os
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import argparse
import time

class BackTranslator:
    def __init__(self, args):
        if args.cuda:
            self.device = torch.device('cuda:' + str(args.gpu_no)) if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        en_to_fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
        self.en_to_fr_tokenizer = MarianTokenizer.from_pretrained(en_to_fr_model_name)
        self.en_to_fr_model = MarianMTModel.from_pretrained(en_to_fr_model_name).to(self.device)

        fr_to_en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
        self.fr_to_en_tokenizer = MarianTokenizer.from_pretrained(fr_to_en_model_name)
        self.fr_to_en_model = MarianMTModel.from_pretrained(fr_to_en_model_name).to(self.device)
  
    def __call__(self, texts):
        with torch.no_grad():
            complete_texts = texts
            backtranslated_texts = []
            for i in range(0, len(texts), 64):
                texts = complete_texts[i: i+64]
                texts = ['>>fr<< ' + text for text in texts]
                en_to_fr_inputs = {key: value.to(self.device) for key, value in self.en_to_fr_tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True).items()}
                pretexts = [self.en_to_fr_tokenizer.decode(text, skip_special_tokens=True) for text in self.en_to_fr_model.generate(**en_to_fr_inputs)]

                texts = ['>>en<< ' + text for text in pretexts]
                fr_to_en_inputs = {key: value.to(self.device) for key, value in self.fr_to_en_tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True).items()}
                texts = [self.fr_to_en_tokenizer.decode(text, skip_special_tokens=True) for text in self.fr_to_en_model.generate(**fr_to_en_inputs)]

                backtranslated_texts.extend(texts)
            return backtranslated_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='The path to LLMBasedArchitecturesForNLI4CT folder')
    parser.add_argument('--cuda', default=False, action='store_true', help='True if using gpu. Default False')
    parser.add_argument('--gpu_no', default=0, type=int, help='The number of gpu to use. Default 0')
    parser.add_argument('--multi_gpu', default=False, action='store_true', help='True if multi-gpu training is to be used. Default False')
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='The gpu ids to use for multi-gpu training. Default "0,1"')
    args = parser.parse_args()
  
    with open(f'{args.root_dir}/Data/train.json', 'r') as file:
        train_data = json.load(file)
    translator = BackTranslator(args)
    aug_data = {}
    for uuid, data_inst in tqdm(train_data.items()):
        if not os.path.exists(f'{args.root_dir}/Data/CTR json/{data_inst["Primary_id"]}_BT.json'):
            with open(f'{args.root_dir}/Data/CTR json/{data_inst["Primary_id"]}.json', 'r') as file:
                data = json.load(file)
            for key in ['Intervention', 'Eligibility', 'Adverse Events', 'Results']:
                data[key] = translator(data[key])
            with open(f'{args.root_dir}/Data/CTR json/{data_inst["Primary_id"]}_BT.json', 'w+') as file:
                json.dump(data, file)
            data_inst["Primary_id"] = data_inst["Primary_id"] + "_BT"
        if ("Secondary_id" in data_inst) and (not os.path.exists(f'{args.root_dir}/Data/CTR json/{data_inst["Secondary_id"]}_BT.json')):
            with open(f'{args.root_dir}/Data/CTR json/{data_inst["Secondary_id"]}.json', 'r') as file:
                data = json.load(file)
            for key in ['Intervention', 'Eligibility', 'Adverse Events', 'Results']:
                data[key] = translator(data[key])
            with open(f'{args.root_dir}/Data/CTR json/{data_inst["Secondary_id"]}_BT.json', 'w+') as file:
                json.dump(data, file)
            data_inst["Secondary_id"] = data_inst["Secondary_id"] + "_BT"
        data_inst["Statement"] = translator([data_inst["Statement"]])[0]
        aug_data[uuid + '_BT'] = data_inst
    for key, value in aug_data.items():
        train_data[key] = value
    with open(f'{args.root_dir}/Data/train.json', 'w+') as file:
        json.dump(train_data, file)
