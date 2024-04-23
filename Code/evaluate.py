import torch
from torchmetrics.classification import BinaryCalibrationError

import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score

def task1_metrics(targets, predictions, uuids, args):
    assert all([(uuid in targets and uuid in predictions) for uuid in uuids])
    true = [targets[uuid]['Label'] == 'Entailment' for uuid in uuids]
    pred = [predictions[uuid]['Prediction'] == 'Entailment' for uuid in uuids]
    prob = [prediction[uuid]['EntailmentProbability'] for uuid in uuids]

    metrics_dict = {}
    metrics_dict['Task1-Entailment-precision'] = precision_score(true, pred)
    metrics_dict['Task1-Entailment-recall'] = recall_score(true, pred)
    metrics_dict['Task1-Entailment-F1'] = f1_score(true, pred)
    metrics_dict['Task1-Contradiction-precision'] = precision_score(1 - true, 1 - pred)
    metrics_dict['Task1-Contradiction-recall'] = recall_score(1 - true, 1 - pred)
    metrics_dict['Task1-Contradiction-F1'] = f1_score(1 - true, 1 - pred)
    metrics_dict['Task1-Macro-F1'] = f1_score(true, pred, average='macro')

    bce_score = BinaryCalibrationError(n_bins=args.n_bins, norm='l1')
    metrics_dict['Task1-Calibration'] = float(bce_score(torch.tensor(prob), torch.tensor(true)).cpu().numpy()[0])
    return metrics_dict

def task2_metrics(targets, predictions, uuids, args):
    TP = 0.0
    FP = 0.0
    FN = 0.0
  
    for uuid in uuids:
        true_evidence_inds = set(targets[uuid]['Primary_evidence_index'])
        predicted_evidence_inds = set(predictions[uuid]['Primary_evidence_index'])
        TP += sum([ind in true_evidence_inds for ind in predictions[uuid]['Primary_evidence_index']])
        FP += sum([ind not in true_evidence_inds for ind in predictions[uuid]['Primary_evidence_index']])
        FN += sum([ind not in predicted_evidence_inds for ind in targets[uuid]['Primary_evidence_index']])
        if targets[uuid]['Type'] == 'Comparison':
            true_evidence_inds = set(targets[uuid]['Secondary_evidence_index'])
            predicted_evidence_inds = set(predictions[uuid]['Secondary_evidence_index'])
            TP += sum([ind in true_evidence_inds for ind in predictions[uuid]['Secondary_evidence_index']])
            FP += sum([ind not in true_evidence_inds for ind in predictions[uuid]['Secondary_evidence_index']])
            FN += sum([ind not in predicted_evidence_inds for ind in targets[uuid]['Secondary_evidence_index']])
    p_score = TP / (TP + FP)
    r_score = TP / (TP + FN)
    F1 = 2 * p_score * r_score / (p_score + r_score)
  
    metrics_dict = {}
    metrics_dict['Task2-precision'] = p_score
    metrics_dict['Task2-recall'] = f_score
    metrics_dict['Task2-F1'] = F1 
    return metrics_dict

def task1_perturbed_metrics(targets, predictions, uuids, args):
    preserving_uuids = [uuid for uuid in uuids if targets[uuid]['Causal_type'][0] == 'Preserving']
    altering_uuids = [uuid for uuid in uuids if targets[uuid]['Causal_type'][0] == 'Altering']

    consistency = sum([predictions[uuid]['Prediction'] == predictions[targets[uuid]['Causal_type'][1]]['Prediction'] for uuids in preserving_uuids]) / len(preserving_uuids)
    faithfulness = sum([predictions[uuid]['Prediction'] != targets[targets[uuid]['Causal_type'][1]]['Label'] for uuids in altering_uuids]) / len(altering_uuids)

    metrics_dict = {}
    metrics_dict['Task1-Consistency'] = consistency
    metrics_dict['Task1-Faithfulness'] = faithfulness
    return metrics_dict

def evaluate_predictions(targets, predictions, args):
    control_set_uuids = [uuid for uuid in targets.keys() if 'Intervention' not in targets[uuid]]
    contrast_set_uuids = [uuid for uuid in targets.keys() if 'Intervention' in targets[uuid]]

    metrics_dict = {}
    if args.evaluate_task1:
        metrics_dict = {**metrics_dict, **task1_metrics(targets, predictions, control_set_uuids, args)}
        if len(contrast_set_uuids):
            metrics_dict = {**metrics_dict, **task1_perturbed_metrics(targets, predictions, contrast_set_uuids, args)}
    if args.evaluate_task2:
        metrics_dict = {**metrics_dict, **task2_metrics(targets, predictions, control_set_uuids, args)}

    return metrics_dict
