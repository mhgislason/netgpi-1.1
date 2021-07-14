
# coding: utf-8
from utility import *
from singlepointernet_model import SinglePointerNet
import copy
import itertools
import pickle
from collections import Counter
from itertools import product
from operator import itemgetter
from sigopt import Connection as soCo
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import time
import sys
from os import path

def train_it(params, train_data):
    criterion = nn.NLLLoss(reduction='sum')
    model = SinglePointerNet(int(params["lstm-hunits"]), int(params["lstm-layers"]), params["lstm-dropout"], int(params["attention-dim"]), True, int(params["embedding-dim"]), 22)
    if USE_CUDA:
        model.cuda()
    
    batch_size = params['batch-size']
    
    optimizer = optim.Adam(model.parameters(), lr=params["learning-rate"], weight_decay=params["weight-decay"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, params['lr-decay-gamma'])

    model.train()
    
    valid_gorodkin = np.zeros(params['epochs'])
    valid_loss = np.zeros(params['epochs'])
    valid_sens = np.zeros(params['epochs'])
    valid_prec = np.zeros(params['epochs'])
    
    dis_train = train_data[(train_data["partition"]!=params["val-partition"])&(train_data["partition"]!=params["test-partition"])]
    dis_valid = train_data[train_data["partition"]==params["val-partition"]]
    
    train_in = get_variable(torch.LongTensor(dis_train["enc_input"].reset_index(drop=True)))
    train_out = get_variable(torch.LongTensor(dis_train["output"].reset_index(drop=True)))
    train_lengths = get_variable(torch.LongTensor(dis_train["seq_length"].reset_index(drop=True)))
    train_kingdom = get_variable(torch.LongTensor(dis_train["enc_kingdom"].reset_index(drop=True)))
    
    valid_in = get_variable(torch.LongTensor(dis_valid["enc_input"].reset_index(drop=True)))
    valid_out = get_variable(torch.LongTensor(dis_valid["output"].reset_index(drop=True)))
    valid_lengths = get_variable(torch.LongTensor(dis_valid["seq_length"].reset_index(drop=True)))
    valid_kingdom = get_variable(torch.LongTensor(dis_valid["enc_kingdom"].reset_index(drop=True)))
    best_model = None
    print ("Train size, Val size:", len(train_in), len(valid_in))  
    for ep in range(params['epochs']):
        
        train_loss, train_accs, train_ratios = [], [], []
        model.train()
        for i, batchdexes in enumerate(np.random.permutation([x for x in get_batch(train_in.shape[0],batch_size)])):
            ## Get training batch
            train_in_batch = train_in[batchdexes[0]:batchdexes[1]]
            train_out_batch = train_out[batchdexes[0]:batchdexes[1]].view(-1,1)
            train_len_batch = train_lengths[batchdexes[0]:batchdexes[1]]
            train_kingd_batch = train_kingdom[batchdexes[0]:batchdexes[1]]

            enc_h = model.init_hidden(train_len_batch.shape[0])
            enc_c = model.init_hidden(train_len_batch.shape[0])
            # if models_df.at[m, 'hidden_kingdom'] > 0:
            #     enc_h = model.init_kingdom(train_len_batch.shape[0], train_kingd_batch)
            # if models_df.at[m, 'hidden_kingdom'] > 1:
            #     enc_c = model.init_kingdom(train_len_batch.shape[0], train_kingd_batch)

            output = model(train_in_batch, enc_h, enc_c, train_len_batch, 0)
            prediction = torch.argmax(output,dim=1).view(-1)
            batch_loss = get_variable(torch.FloatTensor([0.0]))
            for j, o in enumerate(output):
                this_loss = criterion(o[:train_len_batch[j]].view(1,-1), train_out_batch[j])/len(o[:train_len_batch[j]])
#                 print(this_loss)
                batch_loss += this_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_loss.append(get_numpy(batch_loss.squeeze()))
            train_accs.append(np.sum(get_numpy(prediction==train_out_batch.view(-1))) / len(train_in_batch))
            train_ratios.append(len(train_in_batch)/len(train_in))
        
        train_loss = np.array(train_loss)
        train_accs = np.array(train_accs)
        train_ratios = np.array(train_ratios)
        print("train, ep: {} loss: {:.2f} accs: {:.2f}".format(ep, 
                                                            np.mean(train_loss), 
                                                            np.sum(train_accs*train_ratios)))
    
        model.eval()
        last_val_pred = torch.LongTensor(valid_in.shape[0])
        with torch.no_grad():
            val_losses, val_accs, val_ratios = [], [], []
            for val_batchdexes in get_batch(valid_in.shape[0], batch_size):
                ## Get batch
                val_in_batch = valid_in[val_batchdexes[0]:val_batchdexes[1]]
                val_out_batch = valid_out[val_batchdexes[0]:val_batchdexes[1]].view(-1,1)
                val_len_batch = valid_lengths[val_batchdexes[0]:val_batchdexes[1]]
                # val_kingd_batch = valid_kingdom[val_batchdexes[0]:val_batchdexes[1]]
                
                ## Forward pass
                enc_h = model.init_hidden(val_len_batch.shape[0])
                enc_c = model.init_hidden(val_len_batch.shape[0]) # Only used for LSTM
                # if models_df.at[m, 'hidden_kingdom'] > 0:
                #     enc_h = model.init_kingdom(val_len_batch.shape[0], val_kingd_batch)
                # if models_df.at[m, 'hidden_kingdom'] > 1:
                #     enc_c = model.init_kingdom(val_len_batch.shape[0], val_kingd_batch)

                output = model(val_in_batch, enc_h, enc_c, val_len_batch, 0)
                
                val_batch_size = val_in_batch.shape[0]
                
                prediction = torch.argmax(output,dim=1).view(-1)
                last_val_pred[val_batchdexes[0]:val_batchdexes[1]] = prediction
                batch_loss = get_variable(torch.FloatTensor([0.0]))
                for j, o in enumerate(output):
                    this_loss = criterion(o[:val_len_batch[j]].view(1,-1), val_out_batch[j])/len(o[:val_len_batch[j]])
                    batch_loss += this_loss
                # batch_loss = batch_loss / val_batch_size
                
                val_losses.append(get_numpy(batch_loss.squeeze()))
                val_accs.append(np.sum(get_numpy(prediction==val_out_batch.view(-1))) / len(val_in_batch))
                val_ratios.append(val_batch_size/len(valid_in))

            val_ratios = np.array(val_ratios)
            val_loss = np.mean( np.array(val_losses) )
            val_acc = np.sum( np.array(val_accs) * val_ratios )

            gorodkin = matthews_corrcoef(get_numpy(valid_out),get_numpy(last_val_pred))
            valid_gorodkin[ep] = gorodkin
            valid_loss[ep] = val_loss
            
            tp_correctw_count = len([p for t,p,l in zip(get_numpy(valid_out),get_numpy(last_val_pred),get_numpy(valid_lengths)-1) if t<=p+2 and t>=p-2 and t!=l])
            gpi_count = len([t for t,l in zip(get_numpy(valid_out),get_numpy(valid_lengths)-1) if  t!=l])
            pos_pred_count = len([p for p,l in zip(get_numpy(last_val_pred),get_numpy(valid_lengths)-1) if  p!=l])
            sensitivity = tp_correctw_count/gpi_count
            precision = 0.0
            f1_score = 0.0
            if pos_pred_count > 0:
                precision = tp_correctw_count/pos_pred_count
            if sensitivity+precision > 0:
                f1_score = 2 * ( (sensitivity * precision) / (sensitivity + precision) )
            valid_sens[ep] = sensitivity
            valid_prec[ep] = precision
            
            confusius, mcc, auc, t_count, p_count = binarize_presence(get_numpy(valid_out),get_numpy(last_val_pred),get_numpy(valid_lengths))
            
            new_model = {
                "state_dict": model.state_dict(),
                "gorodkin": gorodkin,
                "loss": val_loss,
                "accuracy": val_acc,
                "tp_correctw_count": tp_correctw_count,
                "gpi_count": gpi_count,
                "pos_pred_count": pos_pred_count,
                "sensitivity": sensitivity,
                "precision": precision,
                "f1_score": f1_score,
                "heuristic": 2 * mcc + f1_score,
                "mcc": mcc
            }
            # with open(exec_report_file, 'a+') as outf: 
                    # outf.write("\nepoch:%d, heuristic: %.3f, gorodkin: %.3f\n" % (ep, 2 * mcc + f1_score, gorodkin))
                    # print(new_model,'\n', file=outf)
            if best_model == None:
                best_model = new_model
            elif best_model['heuristic'] < new_model['heuristic']:
                best_model = new_model
            print("val, ep: {} loss: {:.2f} accs: {:.2f}".format(ep, 
                                                    val_loss, 
                                                    val_acc))
            print("Gorodkin:%.3f, MCC:%.3f, f1-score:%.3f, heuristic:%.3f" % 
                (gorodkin,mcc, f1_score,2 * mcc + f1_score))

        scheduler.step()
    return best_model

def main():
    total_runtime = time.time()
    train_data = fetch_data_as_frame(fetching='../../data/short_sequences/gi300_raw_data.fasta', file_type='fasta')
    
    so_connected = soCo(client_token="NNUKWZRYNERBRSYORKHTSXOEOXCZNWOPILUTSLFEOWVVDZVQ")
    

    so_experimental = []
    so_budget = 128
    partitions = set(train_data["partition"].unique())
    test_partitions = partitions
    if len(sys.argv) > 1:
        args = (x for x in sys.argv[1:])
        for arg in args:
            parts = arg
            if ':' in arg:
                parts = arg.split(':')
            elif '=' in arg:
                parts = arg.split('=')
            elif arg in ('-p', '--partitions'):
                try:
                    new_partitions = sorted([int(x) for x in next(args, None).split(',')])
                    test_partitions = new_partitions
                except Exception as e:
                    print (e)
                    return
            elif arg in ('-e', '--experiment-ids'):
                try:
                    e_ids = sorted([int(x) for x in next(args, None).split(',')])
                    so_experimental = [so_connected.experiments(e_id).fetch() for e_id in e_ids]
                except Exception as e:
                    print (e)
                    return
            



    print("Running NetGPI sigopt experiment(s) with test partition %r\n" %(test_partitions))
    exec_report_file = 'pn_sigopt_exec_report_p%s.txt' % (''.join((str(x) for x in test_partitions)))
    
    with open(exec_report_file, 'w+') as outf: 
        print("Running NetGPI sigopt experiment(s) with partitions %r\n" %(test_partitions), file=outf)
    
    for part in test_partitions:
        if part not in partitions:
            raise ValueError("The supplied partitions need to be one or more of %r\n" % (partitions))
            return

    if len(so_experimental) == 0: 
        for part in test_partitions:
            for val_part in partitions:
                if val_part == part:
                    continue
                model_filepath = "../../picklejar/gi300_model_ensemble_tp_%d_vp_%d.pt" % (part,val_part)
                
                if path.exists(model_filepath):
                    saved_model = torch.load(model_filepath)
                    so_experimental.append((part, val_part, saved_model['experiment']))
                else:
                    so_experimental.append((part, val_part, so_connected.experiments().create(
                        name='NetGPI - Max_h - TP:%d,VP:%d - gi300' % (part,val_part),
                        project='netgpi',
                        metrics=[
                            { 'name': 'heuristic',
                            'objective': 'maximize' } ],
                        parameters=[
                            { 'name': 'batch-size', 
                            'type': 'categorical', 
                            'categorical_values': [
                                { "enum_index": 1,
                                    "name": "32" },
                                { "enum_index": 2,
                                    "name": "64" },
                                { "enum_index": 3,
                                    "name": "128" },
                                { "enum_index": 4,
                                    "name": "256" },
                            ] }, 
                            { 'name': 'learning-rate', 
                            'type': 'double', 
                            'bounds': {'min': np.log(0.0001), 'max': np.log(0.02)} },  
                            { 'name': 'lr-decay-gamma', 
                            'type': 'double', 
                            'bounds': {'min': 0.97, 'max': 1} },    
                            { 'name': 'weight-decay', 
                            'type': 'double', 
                            'bounds': {'max': np.log(0.02), 'min': np.log(0.00001)} },
                            { 'name': 'embedding-dim', 
                            'type': 'int', 
                            'bounds': {'min': 12, 'max': 32} }, 
                            { 'name': 'lstm-hunits', 
                            'type': 'int', 
                            'bounds': {'min': 16, 'max': 128} },   
                            { 'name': 'lstm-layers', 
                            'type': 'int', 
                            'bounds': {'min': 3, 'max': 6} },   
                            { 'name': 'lstm-dropout', 
                            'type': 'double', 
                            'bounds': {'min': 0.2, 'max': 0.8} },
                            { 'name': 'attention-dim', 
                            'type': 'int', 
                            'bounds': {'min': 64, 'max': 512} },
                            # { 'name': 'test-partition', 
                            #   'type': 'int', 
                            #   'default_value': part,
                            #   'tunable': False },
                        ],
                        observation_budget=so_budget,
                    ).id))

    suggestions = [{
        "attention-dim": 283,
        "batch-size": "128",
        "embedding-dim": 22,
        "learning-rate": -5.7,
        "lr-decay-gamma": 0.9987,
        "lstm-dropout": 0.6,
        "lstm-hunits": 22,
        "lstm-layers": 4,
        "weight-decay": -4.7
        } for _ in range(0,24)]

    suggestions += [{
        "attention-dim": 283,
        "batch-size": "128",
        "embedding-dim": 16,
        "learning-rate": -5.7,
        "lr-decay-gamma": 0.9987,
        "lstm-dropout": 0.55,
        "lstm-hunits": 16,
        "lstm-layers": 4,
        "weight-decay": -4.7
        } for _ in range(0,24)]

    suggestions += [{
        "attention-dim": 283,
        "batch-size": "128",
        "embedding-dim": 16,
        "learning-rate": -5.7,
        "lr-decay-gamma": 0.9987,
        "lstm-dropout": 0.6,
        "lstm-hunits": 16,
        "lstm-layers": 4,
        "weight-decay": -4.6
        } for _ in range(0,24)]

    suggestions += [{
        "attention-dim": 283,
        "batch-size": "128",
        "embedding-dim": 16,
        "learning-rate": -5.7,
        "lr-decay-gamma": 0.9987,
        "lstm-dropout": 0.6,
        "lstm-hunits": 16,
        "lstm-layers": 4,
        "weight-decay": -4.55
        } for _ in range(0,24)]

    suggestions += [{
        "attention-dim": 340,
        "batch-size": "32",
        "embedding-dim": 16,
        "learning-rate": -6,
        "lr-decay-gamma": 0.997,
        "lstm-dropout": 0.62,
        "lstm-hunits": 20,
        "lstm-layers": 4,
        "weight-decay": -6.25
        } for _ in range(0,24)]

    suggestions += [{
        "attention-dim": 283,
        "batch-size": "128",
        "embedding-dim": 16,
        "learning-rate": -5.412995057916468,
        "lr-decay-gamma": 0.9989309830797753,
        "lstm-dropout": 0.6,
        "lstm-hunits": 16,
        "lstm-layers": 4,
        "weight-decay": -4.605170185988091
        } for _ in range(0,24)]




    for on in range(so_budget):    
        """ For each observation in budget """
        for test_partition, val_partition, experiment_id in so_experimental:
            """ One experiment for each test partition, val partition """
            if time.time() - total_runtime >= 82800:
                ## If runtime >= 23 hours then stop. Continue experiments
                ## using the continue version of the program.
                return
            ## Get a suggestion
            if len(suggestions) == 0:
                return
                suggestion = so_connected.experiments(experiment_id).suggestions().create()
            else:
                suggestion = so_connected.experiments(experiment_id).suggestions().create(
                    assignments = suggestions.pop()
                )

            params = suggestion.assignments
            params['test-partition'] = int(test_partition)
            params['batch-size'] = int(params['batch-size'])
            params["learning-rate"] = np.exp(params["learning-rate"])
            params["weight-decay"] = np.exp(params["weight-decay"])
            params['epochs'] = 300
            ## Run cross validation on the non-test partitions
            ## and collect the heuristic from each fold
            params['val-partition'] = int(val_partition)              
            
            with open(exec_report_file, 'a+') as outf: 
                outf.write("\ntp:%d, vp: %d, on: %d\n" % (test_partition, val_partition, on))
                print(params, file=outf)

            t0 = time.time()
            print(params)
            m = train_it(params, train_data)
            m['val-partition'] = int(val_partition)
            m['test-partition'] = int(test_partition)
            t1 = time.time() - t0
            fold_model = {
                'model':m, 
                'measure': m['heuristic'], 
                'params': params,
                'vp': int(val_partition),
                'tp': int(test_partition),
                'experiment': experiment_id
            }

            with open(exec_report_file, 'a+') as outf: 
                outf.write("\n%.2f\n" % (t1))
                for par in m:
                    if par != 'state_dict':
                        print(par,":",m[par], file=outf)
            
            ## Return the average heuristic to SigOpt
            so_connected.experiments(experiment_id).observations().create(
                suggestion=suggestion.id,
                value=fold_model['measure'],
                value_stddev=0.04
            )
            ## store model if better:
            model_filepath = "../../picklejar/gi300_model_ensemble_tp_%d_vp_%d.pt" % (params['test-partition'],params['val-partition'])
            if not path.exists(model_filepath):
                torch.save(fold_model, model_filepath)
            else:
                saved_model = torch.load(model_filepath)
                if saved_model['measure'] < fold_model['measure']:
                    torch.save(fold_model, model_filepath)
            

if __name__ == "__main__":
    main()
    
