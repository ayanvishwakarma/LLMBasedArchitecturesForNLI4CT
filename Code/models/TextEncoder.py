import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer, AutoConfig, QuantoConfig
from peft import LoraModel, LoraConfig
import numpy as np
import os

class TextEncoder(Module):
    def __init__(self, args, out_size, **kwargs):
        super().__init__(**kwargs)
        self.llm_path = args.llm_path
        self.config = AutoConfig.from_pretrained(self.llm_path)
        if args.quantization is None:
            self.model = AutoModel.from_pretrained(self.llm_path)
        else:
            self.model = AutoModel.from_pretrained(self.llm_path, 
                         quantization_config=QuantoConfig(weights=args.quantization))
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
        self.MAX_SEQ_LEN = args.MAX_SEQ_LEN
        self.device_item = nn.Parameter(torch.tensor(0.0))
        self.linear = nn.Linear(self.config.hidden_size, out_size, 
                                bias=True, dtype=torch.float32)
        
        if args.llm_finetune is False:
            for param in self.model.parameters():
                param.requires_grad = False
        
        if args.use_lora:
            lora_config = LoraConfig(r=args.lora_rank, 
                                     lora_alpha=args.lora_alpha, 
                                     lora_dropout=args.lora_dropout)
            self.model = LoraModel(self.model, 
                                   config=lora_config, 
                                   adapter_name='default')
        
        if args.grad_chkpnt:
            if not self.model.supports_gradient_checkpointing:
                print(f"'{self.llm_path}' does not support gradient checkpointing")
            else:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
                assert self.model.is_gradient_checkpointing

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=[int(x) for x in args.gpu_ids.split(',')])
        
    def forward(self, texts):
        tokenized_texts = self.tokenizer.batch_encode_plus(texts, 
                                                           padding='longest',
                                                           return_tensors='pt',
                                                           return_attention_mask=True,
                                                           return_token_type_ids=True,
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=self.MAX_SEQ_LEN)
        tokenized_texts = {key: value.to(self.device_item.device) for key, value in tokenized_texts.items()}
        output = self.model(**tokenized_texts)
        if type(output) != torch.Tensor:
            output = output.last_hidden_state
        return self.linear(output[:, 0, :])
