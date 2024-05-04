import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import os

from models import TextEncoder

def module_factory(args, comp_name):
    assert (comp_name in ['cross_repr_module', 'entail_head_module'])
    model_name = args.cross_repr_module if comp_name == 'cross_repr_module' else args.entail_head_module
    model_name = model_name.lower()
    if model_name == 'identity':
        return Identity(comp_name, args)
    elif model_name == 'bilstm':
        return BiLSTM(comp_name, args)
    elif model_name == 'transformer':
        return Transformer(comp_name, args)
    elif model_name == 'mean-pooling':
        return MeanPooling(comp_name, args)
    else:
        raise Exception('Model not supported')

class Identity(Module):
    def __init__(self, comp_name, args, **kwargs):
        super().__init__(**kwargs)
        self.comp_name = comp_name
        assert (self.comp_name in ['cross_repr_module', 'entail_head_module'])
        self.evidence_classify = args.evidence_classify
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, 1, 
                                        bias=True, dtype=torch.float32),
                                        nn.Sigmoid())
        
    def forward(self, inputs):
        prob = self.classifier(inputs).squeeze(-1)
        if self.comp_name == 'entail_head_module':
            return prob[0]
        else:
            return inputs, prob[1:]
        
class BiLSTM(Module):
    def __init__(self, comp_name, args, **kwargs):
        super().__init__(**kwargs)
        self.comp_name = comp_name
        assert (self.comp_name in ['cross_repr_module', 'entail_head_module'])
        self.evidence_classify = args.evidence_classify
        if self.comp_name == 'cross_repr_module':
            num_layers = args.num_layers_cross_repr
        else:
            num_layers = args.num_layers_entail_head
        self.encoder = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size,
                               num_layers=num_layers, bias=True, dropout=args.dropout,
                               bidirectional=True, dtype=torch.float32)
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, 1, 
                                        bias=True, dtype=torch.float32),
                                        nn.Sigmoid())
        
    def forward(self, inputs):
        encoded_inputs = self.encoder(inputs)
        prob = self.classifier(encoded_inputs if self.evidence_classify == 'post' else inputs).squeeze(-1)
        if self.comp_name == 'entail_head_module':
            return prob[0]
        else:
            return encoded_inputs, prob[1:]
    
class Transformer(Module):
    def __init__(self, comp_name, args, **kwargs):
        super().__init__(**kwargs)
        self.comp_name = comp_name
        assert (self.comp_name in ['cross_repr_module', 'entail_head_module'])
        self.evidence_classify = args.evidence_classify
        if self.comp_name == 'cross_repr_module':
            num_layers = args.num_layers_cross_repr
        else:
            num_layers = args.num_layers_entail_head
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_size, 
                                             nhead=args.n_heads, dim_feedforward=args.ff_dim,
                                             dropout=args.dropout, dtype=torch.float32), 
                                             num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, 1, 
                                        bias=True, dtype=torch.float32),
                                        nn.Sigmoid())
        
    def forward(self, inputs):
        encoded_inputs = self.encoder(inputs)
        prob = self.classifier(encoded_inputs if self.evidence_classify == 'post' else inputs).squeeze(-1)
        if self.comp_name == 'entail_head_module':
            return prob[0]
        else:
            return encoded_inputs, prob[1:]

class MeanPooling(Module):
    def __init__(self, comp_name, args, **kwargs):
        super().__init__(**kwargs)
        self.comp_name = comp_name
        assert (self.comp_name in ['entail_head_module'])
        self.evidence_classify = args.evidence_classify
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, 1, 
                                        bias=True, dtype=torch.float32),
                                        nn.Sigmoid())
        
    def forward(self, inputs):
        return self.classifier(inputs.mean(dim=0).unsqueeze(0)).squeeze(0)[0]
        
class ModelArchitecture1(Module):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.device_item = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.text_encoder = TextEncoder(args, args.hidden_size)
        self.CLS_EMBD = nn.Parameter(torch.zeros(args.hidden_size, dtype=torch.float32))
        self.cross_repr_module = module_factory(args, 'cross_repr_module')
        self.register_buffer('thresh_evidence', torch.tensor(0.0, dtype=torch.float32))
        self.entail_head_module = module_factory(args, 'entail_head_module')
        self.register_buffer('thresh_entailment', torch.tensor(0.0, dtype=torch.float32))

        if self.args.pos_emb == 'learnable':
            self.pos_weights = nn.Linear(1, args.hidden_size, bias=True, dtype=torch.float32)
        
    def positional_embedding(self, pos_ids):
        device = pos_ids.device
        if self.args.pos_emb == 'static':
            w = pos_ids.unsqueeze(-1) / (10000 ** torch.linspace(0, 1, self.args.hidden_size//2).to(device))
            return torch.cat([torch.sin(w), torch.cos(w)], dim=-1)
        elif self.args.pos_emb == 'learnable':
            w = self.pos_weights(pos_ids.unsqueeze(-1))
            w = torch.cat([w[:, :1], torch.sin(w[:, 1:])], dim=-1)
            return w
        
    def forward(self, data_dict):
        device = self.device_item.device
        text_input = [f"The Hypothesis to be evaluated for 'Entailment | Contradiction' is '{data_dict['hypothesis']}'"] \
                     + [(f'Hypothesis: "{data_dict["hypothesis"]}"[SEP]Premise: {line}' if self.args.prepend_hypot else line) for line in data_dict['premises']]
        text_embed = self.text_encoder(text_input)
        text_embed[0] += self.CLS_EMBD

        if self.args.pos_emb is not None:
            pos_emb = self.positional_embedding(torch.arange(text_embed.shape[0], dtype=torch.float32).to(device))
            text_embed += pos_emb
        
        cross_repr_output, evidence_prob = self.cross_repr_module(text_embed)
        if not self.args.ignore_evidence_selection:
            if self.training:
                evidence_labels = torch.tensor(data_dict['label_task2'])
                evidence_inds = torch.where(evidence_labels)[0].to(device)
            else:
                evidence_inds = torch.where(evidence_prob >= self.thresh_evidence)[0].to(device)
            entail_head_input = cross_repr_output[torch.cat([torch.tensor([0]).to(device), evidence_inds], dim=-1)]
        else:
            entail_head_input = cross_repr_output
        
        entailment_prob = self.entail_head_module(entail_head_input)
        
        return entailment_prob, evidence_prob

    def get_predictions(self, entailment_prob, evidence_prob):
        return entailment_prob >= self.thresh_entailment, evidence_prob >= self.thresh_evidence
        
    def on_train_epoch_end(self, entailment_labels, entailment_logits, evidence_labels, evidence_logits, task1_monitor='Macro-F1'):
        # Adjust thresholds so maximize F1-score for task1 and task2.
        device = self.device_item.device
        sorted_inds = torch.argsort(torch.tensor(entailment_logits).to(device))
        entailment_labels = torch.tensor(entailment_labels, dtype=torch.int32).to(device)[sorted_inds]
        entailment_logits = torch.tensor(entailment_logits, dtype=torch.float32).to(device)[sorted_inds]
        
        thresholds = (entailment_logits[:-1] + entailment_logits[1:]) / 2
        TP = torch.flip(torch.cumsum(torch.flip(entailment_labels, dims=(-1,))[:-1] == 1, dim=-1), dims=(-1,))
        FP = torch.flip(torch.cumsum(torch.flip(entailment_labels, dims=(-1,))[:-1] == 0, dim=-1), dims=(-1,))
        TN = torch.cumsum(entailment_labels[:-1] == 0, dim=-1)
        FN = torch.cumsum(entailment_labels[:-1] == 1, dim=-1)
        
        precision_entailment = TP / (TP + FP + 1e-8)
        recall_entailment = TP / (TP + FN + 1e-8)
        F1_entailment = 2 * precision_entailment * recall_entailment / (precision_entailment + recall_entailment + 1e-8)
        
        precision_contradiction = TN / (TN + FN + 1e-8)
        recall_contradiction = TN / (TN + FP + 1e-8)
        F1_contradiction = 2 * precision_contradiction * recall_contradiction / (precision_contradiction + recall_contradiction + 1e-8)
        
        macro_F1 = (F1_entailment + F1_contradiction) / 2
        if task1_monitor == 'Macro-F1':
            index = torch.argmax(macro_F1)
        elif task1_monitor == 'F1-entail':
            index = torch.argmax(F1_entailment)
        else:
            raise Exception("task1 monitor quantity should be either Macro-F1 or F1-entail")
        self.register_buffer('thresh_entailment', thresholds[index])
        
        sorted_inds = torch.argsort(torch.tensor(evidence_logits).to(device))
        evidence_labels = torch.tensor(evidence_labels, dtype=torch.int32).to(device)[sorted_inds]
        evidence_logits = torch.tensor(evidence_logits, dtype=torch.float32).to(device)[sorted_inds]
        
        thresholds = (evidence_logits[:-1] + evidence_logits[1:]) / 2
        TP = torch.flip(torch.cumsum(torch.flip(evidence_logits, dims=(-1,))[:-1] == 1, dim=-1), dims=(-1,))
        FP = torch.flip(torch.cumsum(torch.flip(evidence_logits, dims=(-1,))[:-1] == 0, dim=-1), dims=(-1,))
        TN = torch.cumsum(evidence_logits[:-1] == 0, dim=-1)
        FN = torch.cumsum(evidence_logits[:-1] == 1, dim=-1)
        
        precision_evidence = TP / (TP + FP + 1e-8)
        recall_evidence = TP / (TP + FN + 1e-8)
        F1_evidence = 2 * precision_evidence * recall_evidence / (precision_evidence + recall_evidence + 1e-8)
        
        self.register_buffer('thresh_evidence', thresholds[torch.argmax(F1_evidence)])
