import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import torch.optim as optim
import pickle
import collections
import time
import os
import argparse

from data import DatasetNLI4CT
from evaluate import evaluate_predictions

def get_loss_fn(args):
    def loss_fn(pred_task1, true_task1, pred_task2, true_task2):
        if args.loss == 'ce':
            return args.Lambda * nn.BCELoss(pred_task1, true_task1) + (1.0 - args.Lambda) * nn.BCELoss(pred_task2, true_task2)
        # elif args.loss == 'focal':
        #     pass
    return loss_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument('--exp-name', default='exp-0', type=str, help='The name of experiment')
    parser.add_argument('--seed', default=2024, type=int, help='The seed value for reproducibility. Default 2024')

    # Dataset args
    parser.add_argument('--data_ablation', default=None, type=str, help='The data ablation study to consider. Default None',
                        choices=[None, 'hypothesis-only', 'gold-evidence'])
    
    # Text Enocder args
    parser.add_argument('--llm_path', default='microsoft/deberta-base', type=str, 
                        help='The path to hugging-face llm model. Default microsoft/deberta-base')
    parser.add_argument('--MAX_SEQ_LEN', default=512, type=int, 
                        help='The max-sequence length for llm tokenizer. Default 512')
    parser.add_argument('--llm_finetune', action='store_true', help='Finetune llm. Default False')
    parser.add_argument('--use_lora', action='store_true', help='. Default False')
    parser.add_argument('--grad_chkpnt', action='store_true', help='Use Gradient Checkpointing. Default False')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs. Default False')
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='The gpu id\'s to use. Default 0,1')
    
    # Model Architecture args
    parser.add_argument('--head1', default='transformer', type=str, help='The head1 model to consider. Default transformer',
                        choices=['identity', 'transformer', 'bilstm'])
    parser.add_argument('--head2', default='identity', type=str, help='The head2 model to consider. Default identity',
                        choices=['identity', 'transformer', 'bilstm'])
    parser.add_argument('--hidden_size', default=128, type=int, help='The dimension of hidden layers of model. Default 128')
    parser.add_argument('--num_layers_head1', default=4, type=int, help='The number of layers in head1. Default 4')
    parser.add_argument('--num_layers_head2', default=4, type=int, help='The number of layers in head2. Default 4')
    parser.add_argument('--dropout', default=0.2, type=float, help='The dropout rate in head1 and head2. Default 0.2')
    parser.add_argument('--pos_emb', default=None, type=str, help='The positional embedding to use in text-embedding output. Default None',
                        choices=[None, 'static', 'learnable'])
    
    # Evaluation metric args
    parser.add_argument('--evaluate_task1', action='store_true', help='Evaluate metrics for task 1. Default False')
    parser.add_argument('--evaluate_task2', action='store_true', help='Evaluate metrics for task 2. Default False')
    parser.add_argument('--n_bins', default=10, type=int, help='The number of bins to use in ECE-calibration metric. Default 10')
    
    # Training args
    parser.add_argument('--loss', default='ce', type=str, help='The loss function to use. Default ce(cross-entropy)', choices=['ce', 'focal'])
    parser.add_argument('--Lambda', default=0.5, type=float, help='The lambda value for task1/task2 loss aggregation. Default 0.5')
    parser.add_argument('--lr', default=0.0005, type=float, help='The learning rate. Default 0.0005')
    parser.add_argument('--bs', default=32, type=int, help='The batch size. Default 32')
    parser.add_argument('--epochs', default=8, type=int, help='Number of epochs to run the model. Default 8')
    parser.add_argument('--cuda', default=False, action=argparse.BooleanOptionalAction, help='True if using gpu. Default False')
    parser.add_argument('--gpu_no', default=0, type=int, help='The gpu id to use. Default 0')
    parser.add_argument('--patience_es', default=5, type=int, help='Patience of early stopping. Default 5')
    parser.add_argument('--delta_es', default=0.0, type=float, help='Delta of early stopping. Default 0.0')
    parser.add_argument('--scheduler', default=0.5, type=float, help='Threshold for early stopping. Default 0.5')
  
    args = parser.parse_args()
    assert(0.0 <= args.Lambda <= 1.0)
    
    # ------------------------------Result Address------------------------------
    root_dir = __file__
    result_addr = f'{root_dir}/Results/{args.exp_name}-{str(args.seed)}'

    # ------------------------------Parameters to save------------------------------
    train_epoch_loss = []
    train_task1_F1 = []
    train_task2_F1 = []
    train_faithfulness = []
    train_consistency = []
    val_epoch_loss = []
    val_task1_F1 = []
    val_task2_F1 = []
    val_faithfulness = []
    val_consistency = []
    epoch_time = []

    # ------------------------------Prepare DataLoaders------------------------------
    seed_everything(seed)
    if args.cuda:
        device = torch.device('cuda:' + str(gpu_no)) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    print("device: ", device)
    trainset = DatasetNLI4CT(root_dir=root_dir, split_name='train', args=args)
    devset = DatasetNLI4CT(root_dir=root_dir, split_name='dev', args=args)
    testset = DatasetNLI4CT(root_dir=root_dir, split_name='test', args=args)

    # ------------------------------Initialize early stopping------------------------------
    early_stopping = EarlyStopping(patience=args.patience_es, verbose=True, delta=args.delta_es, 
                                   save_path=os.path.join(result_addr, 'model_state_dict.pt'))

    # ------------------------------Model Creation------------------------------
    model = ModelArchitecture1(args)
    model.to(device)
    loss_fn = get_loss_fn(args)
    optimizer = optim.AdamW([{"params": [p for n, p in model.named_parameters() if 'text_encoder' in n], "weight_decay_rate": 0.01},
                             {"params": [p for n, p in model.named_parameters() if 'text_encoder' not in n]}], lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor,
                                                     patience=1, threshold=0.0001, threshold_mode='rel',
                                                     cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    # ------------------------------Model Training------------------------------
    for e in range(args.epochs):
        print("Epoch: ", e+1)
        train_loss = 0
        train_pred = {}
        val_loss = 0
        val_pred = {}

        train_task1_labels = []
        train_task1_logits = []
        train_task2_labels = []
        train_task2_logits = []
        
        model.train()
        st_time = time.time()
        batch_processed = 0
        for sample in tqdm(trainset):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                entailment_prob, evidence_prob, entailment_pred, evidence_pred = model.forward(sample)
                loss = (1 / args.batch_size) * loss_fn((entailment_prob, torch.tensor([sample['label_task1']]), 
                                                        evidence_prob, torch.tensor(sample['label_task2'])))
            scaler.scale(loss).backward()
            batch_processed = (batch_processed + 1) % batch_size
            if batch_processed == 0:
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            train_loss = train_loss + loss.item()
            train_pred[sample['uuid']] = {'Prediction': 'Entailment' if entailment_pred else 'Contradiction',
                                          'EntailmentProbability': float(entailment_prob.cpu().numpy()[0]),
                                          'Primary_evidence_index': [int(i) for i, x, y in enumerate(zip(entailment_pred, sample['premise_ids'])) 
                                                                     if y == 1 and x == 1],
                                          'Primary_evidence_prob': [float(x) for x, y in zip(evidence_prob, sample['premise_ids']) if y == 1]}
            if sample['type'] == 'Comparison':
                offset =  sum([1 if x == 1 else 0 for x in sample['premise_ids']])
                train_pred[sample['uuid']]['Secondary_evidence_index'] = [int(i) - offset for i, x, y in enumerate(zip(entailment_pred, sample['premise_ids'])) 
                                                                          if y == 2 and x == 1]
                train_pred[sample['uuid']]['Secondary_evidence_prob'] = [float(x) for x, y in zip(evidence_prob, sample['premise_ids']) if y == 2]
            train_task1_labels.append(sample['label_task1'])
            train_task1_logits.extend([float(x) for x in entailment_prob.cpu().numpy()])
            train_task2_labels.extend(sample['label_task2'])
            train_task2_logits.extend([float(x) for x in evidence_prob.cpu().numpy()])
        model.on_train_epoch_end(train_task1_labels, train_task1_logits, train_task2_labels, train_task2_logits)
        end_time = time.time()
        epoch_time.append(end_time - st_time)
        print("Epoch time: ", epoch_time[e])

        model.eval()
        for sample in tqdm(devset):
            with torch.no_grad():
                entailment_prob, evidence_prob, entailment_pred, evidence_pred = model.forward(sample)
                loss = (1 / args.batch_size) * loss_fn((entailment_prob, torch.tensor([sample['label_task1']]), 
                                                        evidence_prob, torch.tensor(sample['label_task2'])))
            val_loss = val_loss + loss.item()
            val_pred[sample['uuid']] = {'Prediction': 'Entailment' if entailment_pred else 'Contradiction',
                                        'EntailmentProbability': float(entailment_prob.cpu().numpy()[0]),
                                        'Primary_evidence_index': [int(i) for i, x, y in enumerate(zip(entailment_pred, sample['premise_ids'])) 
                                                                   if y == 1 and x == 1],
                                        'Primary_evidence_prob': [float(x) for x, y in zip(evidence_prob, sample['premise_ids']) if y == 1]}
            if sample['type'] == 'Comparison':
                offset =  sum([1 if x == 1 else 0 for x in sample['premise_ids']])
                val_pred[sample['uuid']]['Secondary_evidence_index'] = [int(i) - offset for i, x, y in enumerate(zip(entailment_pred, sample['premise_ids'])) 
                                                                        if y == 2 and x == 1]
                val_pred[sample['uuid']]['Secondary_evidence_prob'] = [float(x) for x, y in zip(evidence_prob, sample['premise_ids']) if y == 2]

        # Calculate mean loss of training data and validation data
        train_epoch_loss.append(train_loss/len(trainset))
        val_epoch_loss.append(val_loss/len(devset))

        # Metrics
        with open(os.path.join(root_dir, f'Data/train.json'), 'r') as file:
            targets = json.load(file)
        train_metrics = evaluate_predictions(targets, train_pred, args)
        with open(os.path.join(root_dir, f'Data/dev.json'), 'r') as file:
            targets = json.load(file)
        val_metrics = evaluate_predictions(targets, val_pred, args)
        
        train_task1_F1.append(train_metrics['Task1-Macro-F1'])
        train_task2_F1.append(train_metrics['Task2-F1'])
        train_faithfulness.append(train_metrics['Task1-Consistency'])
        train_consistency.append(train_metrics['Task1-Faithfulness'])
        
        val_task1_F1.append(val_metrics['Task1-Macro-F1'])
        val_task2_F1.append(val_metrics['Task2-F1'])
        val_faithfulness.append(val_metrics['Task1-Consistency'])
        val_consistency.append(val_metrics['Task1-Faithfulness'])      
        
        print("Train Loss: ", train_epoch_loss[e], "\t Val Loss: ", val_epoch_loss[e])
        print("Train Task1-Macro-F1: ", train_metrics['Task1-Macro-F1'], "\t Val Task1-Macro-F1: ", val_metrics['Task1-Macro-F1'])
        print("Train Task2-F1: ", train_metrics['Task2-F1'], "\t Val Task2-F1: ", val_metrics['Task2-F1'])
        print("Train Task1-Consistency: ", train_metrics['Task1-Consistency'], "\t Val Task1-Consistency: ", val_metrics['Task1-Consistency'])
        print("Train Task1-Faithfulness: ", train_metrics['Task1-Faithfulness'], "\t Val Task1-Faithfulness: ", val_metrics['Task1-Faithfulness'])

        # early stopping
        early_stopping(val_metrics['Task1-Macro-F1'], model)
        if early_stopping.early_stop:
            print(f"Early Stopping after {e+1} epochs")
            break    
        scheduler.step(val_metrics['Task1-Macro-F1'])

    # ------------------------------Save result on train and val data------------------------------
    result = {'args': args,
              
              'train_epoch_loss': train_epoch_loss,
              'train_task1_F1': train_task1_F1,
              'train_task2_F1': train_task2_F1,
              'train_faithfulness': train_faithfulness,
              'train_consistency': train_consistency,
              
              'val_epoch_loss': val_epoch_loss,
              'val_task1_F1': val_task1_F1,
              'val_task2_F1': val_task2_F1,
              'val_faithfulness': val_faithfulness,
              'val_consistency': val_consistency,
              
              'epoch_time': epoch_time}

    # ------------------------------Load model for testing------------------------------
    best_model_auprc = ModelArchitecture1(args)
    best_model_auprc.load_state_dict(os.path.join(result_addr, 'model_state_dict.pt'))
    print("Model based on AUPRC loaded for testing.")
    
    train_loss = 0
    train_pred = {}
    val_loss = 0
    val_pred = {}
    test_loss = 0
    test_pred = {}
    
    best_model_auprc.eval()
    for dataset, split_name in [(trainset, 'train'), (devset, 'dev'), (testset, 'test')]:
        if split_name == 'train':
            pred_dict = train_pred
        elif split_name == 'dev':
            pred_dict = val_pred
        elif split_name == 'test':
            pred_dict = test_pred
            
        for sample in tqdm(devset):
            with torch.no_grad():
                entailment_prob, evidence_prob = model.forward(sample)
                loss = (1 / args.batch_size) * loss_fn((entailment_prob, torch.tensor([sample['label_task1']]), 
                                                        evidence_prob, torch.tensor(sample['label_task2'])))
            if split_name == 'test':
                test_loss = test_loss + loss.item()
            pred_dict[sample['uuid']] = {'Prediction': 'Entailment' if entailment_pred else 'Contradiction',
                                         'EntailmentProbability': float(entailment_prob.cpu().numpy()[0]),
                                         'Primary_evidence_index': [int(i) for i, x, y in enumerate(zip(entailment_pred, sample['premise_ids'])) 
                                                                    if y == 1 and x == 1],
                                         'Primary_evidence_prob': [float(x) for x, y in zip(evidence_prob, sample['premise_ids']) if y == 1]}
            if sample['type'] == 'Comparison':
                offset =  sum([1 if x == 1 else 0 for x in sample['premise_ids']])
                pred_dict[sample['uuid']]['Secondary_evidence_index'] = [int(i) - offset for i, x, y in enumerate(zip(entailment_pred, sample['premise_ids'])) 
                                                                        if y == 2 and x == 1]
                pred_dict[sample['uuid']]['Secondary_evidence_prob'] = [float(x) for x, y in zip(evidence_prob, sample['premise_ids']) if y == 2]
                
        with open(os.path.join(root_dir, f'Data/{split_name}.json'), 'r') as file:
            targets = json.load(file)
        metrics = evaluate_predictions(targets, pred_dict, args)
        result[f'best-model-{split_name}-metrics'] = metrics
                        
        with open(os.path.join(result_addr, f'{split_name}.json'), 'w') as file:
            targets = json.dump(pred_dict, file)

    result['test_loss'] = test_loss/len(testset)   

    # ------------------------------Save results to a file------------------------------
    with open(os.path.join(result_addr, 'results.data'), 'wb') as file:
            pickle.dump(result, file)

    print("Model succefully run and results and model are stored")