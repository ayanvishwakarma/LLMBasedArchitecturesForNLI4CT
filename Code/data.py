import torch
from torch.utils.data import Dataset
import numpy as np
import json

class DatasetNLI4CT(Dataset):
    def __init__(self, root_dir, split_name, args, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = __file__ if root_dir is None else root_dir
        
        with open(f'{self.root_dir}/Data/{split_name}.json', 'r') as file:
            self.data = json.load(file)
        self.uuids = list(self.data.keys())
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
            texts.extend([f'Hypothesis: "{data_inst["Statement"]}", Premise: In Primary CTR {data_inst["Section_id"]} section, ' 
                          + (f'under subsection "{section_text[subsection_ids[i]]}", ' if subsection_ids[i] >= 0 else '') 
                          + f'in line {i} the following text is written "{section_text[i]}"' for i in range(len(section_text))])
            text_ids.extend([1 for i in range(len(section_text))])
            evidence_inds = set(data_inst['Primary_evidence_index'])
            labels_task2.extend([int(i in evidence_inds) for i in range(len(section_text))])
            
        if data_inst['Type'] == 'Comparison':
            with open(f'{self.root_dir}/Data/CTR json/{data_inst["Secondary_id"]}.json', 'r') as file:
                ctr = json.load(file)
                section_text = ctr[data_inst['Section_id']]
                subsection_ids = np.maximum.accumulate([(-1 if x.startswith(' ') else i) for i, x in enumerate(section_text)])
                texts.extend([f'Hypothesis: "{data_inst["Statement"]}", Premise: In Secondary CTR {data_inst["Section_id"]} section, ' 
                              + (f'under subsection "{section_text[subsection_ids[i]]}", ' if subsection_ids[i] >= 0 else '')
                              + f'in line {i} the following text is written "{section_text[i]}"' for i in range(len(section_text))])
                text_ids.extend([2 for i in range(len(section_text))])
                evidence_inds = set(data_inst['Secondary_evidence_index'])
                labels_task2.extend([int(i in evidence_inds) for i in range(len(section_text))])
        
        ouput_data = {'uuid': uuid,
                      'hypothesis': data_inst['Statement'],
                      'premises': texts,
                      'premise_ids': text_ids,
                      'label_task1': int(data_inst['Label'] == 'Entailment'),
                      'label_task2': labels_task2} 
        return ouput_data
