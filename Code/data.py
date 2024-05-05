import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

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
        texts = ['>>fr<< ' + text for text in texts]
        en_to_fr_inputs = {key: value.to(self.device) for key, value in self.en_to_fr_tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True).items()}
        pretexts = [self.en_to_fr_tokenizer.decode(text, skip_special_tokens=True) for text in self.en_to_fr_model.generate(**en_to_fr_inputs)]

        texts = ['>>en<< ' + text for text in pretexts]
        fr_to_en_inputs = {key: value.to(self.device) for key, value in self.fr_to_en_tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True).items()}
        texts = [self.fr_to_en_tokenizer.decode(text, skip_special_tokens=True) for text in self.fr_to_en_model.generate(**fr_to_en_inputs)]

        return texts

class DatasetNLI4CT(Dataset):
    def __init__(self, root_dir, split_name, args, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.data_ablation = args.data_ablation
        self.root_dir = __file__ if root_dir is None else root_dir
        
        with open(f'{self.root_dir}/Data/{split_name}.json', 'r') as file:
            self.data = json.load(file)

        if split_name == 'train' and args.backtranslate:
            translator = BackTranslator(args)
            for uuid, data_inst in self.data.items():
                if not os.path.exists(f'{self.root_dir}/Data/CTR json/{data_inst["Primary_id"]}_BT.json'):
                    with open(f'{self.root_dir}/Data/CTR json/{data_inst["Primary_id"]}.json', 'r') as file:
                        data = json.load(file)
                    
            self.data.keys()
            
        
        self.uuids = list(self.data.keys())
        if verbose:
            print(f'Number of instances in {split_name}: {len(self.uuids)}')
    
    def __len__(self):
        return len(self.uuids)
    
    def __getitem__(self, index):
        uuid = self.uuids[index]
        data_inst = self.data[uuid]
        texts = []
        text_ids = []
        labels_task2 = []
        
        with open(f'{self.root_dir}/Data/CTR json/{data_inst["Primary_id"]}.json', 'r') as file:
            ctr = json.load(file)
            section_text = ctr[data_inst['Section_id']]
            subsection_ids = np.maximum.accumulate([(-1 if x.startswith(' ') else i) for i, x in enumerate(section_text)])
            texts.extend([f'In Primary CTR {data_inst["Section_id"]} section, ' 
                          + (f'under subsection "{section_text[subsection_ids[i]]}", ' if subsection_ids[i] >= 0 else '') 
                          + f'in line {i} the following premise is given "{section_text[i]}"' for i in range(len(section_text))])
            text_ids.extend([1 for i in range(len(section_text))])
            if 'Primary_evidence_index' in data_inst:
                evidence_inds = set(data_inst['Primary_evidence_index'])
                labels_task2.extend([int(i in evidence_inds) for i in range(len(section_text))])
            else:
                labels_task2.extend([int(-1) for i in range(len(section_text))])
            
        if data_inst['Type'] == 'Comparison':
            with open(f'{self.root_dir}/Data/CTR json/{data_inst["Secondary_id"]}.json', 'r') as file:
                ctr = json.load(file)
                section_text = ctr[data_inst['Section_id']]
                subsection_ids = np.maximum.accumulate([(-1 if x.startswith(' ') else i) for i, x in enumerate(section_text)])
                texts.extend([f'In Secondary CTR {data_inst["Section_id"]} section, ' 
                              + (f'under subsection "{section_text[subsection_ids[i]]}", ' if subsection_ids[i] >= 0 else '')
                              + f'in line {i} the following premise is given "{section_text[i]}"' for i in range(len(section_text))])
                text_ids.extend([2 for i in range(len(section_text))])
                if 'Secondary_evidence_index' in data_inst:
                    evidence_inds = set(data_inst['Secondary_evidence_index'])
                    labels_task2.extend([int(i in evidence_inds) for i in range(len(section_text))])
                else:
                    labels_task2.extend([-1 for i in range(len(section_text))])
                    
        if self.data_ablation == 'hypothesis-only':
            texts = []
            text_ids = []
            labels_task2 = []
        elif self.data_ablation == 'gold-evidence':
            texts = [texts[i] for i, x in range(len(labels_task2)) if x]
            text_ids = [text_ids[i] for i, x in range(len(labels_task2)) if x]
            labels_task2 = [labels_task2[i] for i, x in range(len(labels_task2)) if x] 
        
        output_data = {'uuid': uuid,
                      'type': data_inst['Type'],
                      'hypothesis': data_inst['Statement'],
                      'premises': texts,
                      'premise_ids': text_ids,
                      'label_task1': int(data_inst['Label'] == 'Entailment'),
                      'label_task2': [int(x) for x in labels_task2]} 
        return output_data
