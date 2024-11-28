import copy
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from torch import optim
from model.graph_pool import CategoricalGraphAtt
# Update the import path to match your project structure
from parse_arg import parse_basic_args    

args = parse_basic_args()
print(args)
data_path = args.data

with open(data_path, "rb") as f:
    data = pickle.load(f)
inner_edge = np.array(np.load("D:\\capstone_project\\capstone_project\\inner.npy"))
outer_edge = np.array(np.load("D:\\capstone_project\\capstone_project\\outer.npy"))
time_step = data["train"]["x1"].shape[-2]
input_dim = data["train"]["x1"].shape[-1]
num_weeks = data["train"]["x1"].shape[0]
train_size = int(num_weeks * 0.2)
device = args.device
agg_week_num = args.week_num

# Convert data into torch dtype
train_w1 = torch.Tensor(data["train"]["x1"].astype(float)).float().to(device)
train_w2 = torch.Tensor(data["train"]["x2"].astype(float)).float().to(device)
train_w3 = torch.Tensor(data["train"]["x3"].astype(float)).float().to(device)
train_w4 = torch.Tensor(data["train"]["x4"].astype(float)).float().to(device)
inner_edge = torch.tensor(inner_edge.T, dtype=torch.int64).to(device)
outer_edge = torch.tensor(outer_edge.T, dtype=torch.int64).to(device)

# Test data
test_w1 = torch.Tensor(data["test"]["x1"].astype(float)).float().to(device)
test_w2 = torch.Tensor(data["test"]["x2"].astype(float)).float().to(device)
test_w3 = torch.Tensor(data["test"]["x3"].astype(float)).float().to(device)
test_w4 = torch.Tensor(data["test"]["x4"].astype(float)).float().to(device)
test_data = [test_w1, test_w2, test_w3, test_w4]

# Label data
train_reg = torch.Tensor(data["train"]["y_return_ratio"].astype(float)).float()
train_cls = torch.Tensor(data["train"]["y_up_or_down"].astype(float)).float()
test_y = data["test"]["y_return_ratio"]
test_cls = data["test"]["y_up_or_down"]
test_shape = test_y.shape[0]
loop_number = 100
ks_list = [5, 10, 20]

def MRR(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict = predict.sort_values("pred_y", ascending=False).reset_index(drop=True)
    predict["pred_y_rank_index"] = (predict.index) + 1
    predict = predict.sort_values("y", ascending=False)

    return sum(1 / predict["pred_y_rank_index"][:k])


def Precision(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict1 = predict.sort_values("pred_y", ascending=False)
    predict2 = predict.sort_values("y", ascending=False)
    correct = len(list(set(predict1["y"][:k].index) & set(predict2["y"][:k].index)))
    return correct / k


def IRR(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict1 = predict.sort_values("pred_y", ascending=False)
    predict2 = predict.sort_values("y", ascending=False)
    return sum(predict2["y"][:k]) - sum(predict1["y"][:k])


def Acc(test_y, pred_y):
    test_y = np.ravel(test_y)
    pred_y = np.ravel(pred_y)
    pred_y = (pred_y > 0) * 1
    pred_y = pred_y[:18050]
    sum = 0
    for i in range(len(pred_y)):
        if(test_y[i]==pred_y[i]):
            sum += 1
    acc_score = sum / len(pred_y)

    return acc_score

def top_stocks(pred, top_k=5):
    NIFTY50_name = pd.read_csv("D:\\capstone_project\\capstone_project\\NIFTY50_category.csv")
    with_indices = []
    for i in range(18000, 18050):
        with_indices.append([pred[i], i-18000])
    
    with_indices = sorted(with_indices)
    print("The top", top_k, "Stocks are: ")
    print("-----------------------------------")
    for i in range (top_k):
        print(NIFTY50_name.iloc[with_indices[i][1]]['company'], ", Sector:", NIFTY50_name.iloc[with_indices[i][1]]['category'])

def train(args):
    global predictions
    global test_y
    model_name = args.model
    l2 = args.l2
    lr = args.lr
    beta = args.beta
    gamma = args.gamma 
    alpha = args.alpha
    device = args.device
    epochs = args.epochs
    hidden_dim = args.dim 
    use_gru = args.use_gru

    if model_name == "CAT":
        model = CategoricalGraphAtt(input_dim,time_step,hidden_dim,inner_edge,outer_edge,agg_week_num,use_gru,device).to(device)

    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:%s" % pytorch_total_params)
    print("-----------------------------------------------")

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), weight_decay=l2,lr=lr)
    reg_loss_func = nn.L1Loss(reduction='none')
    cls_loss_func = nn.BCELoss(reduction='none')

    # Save best model
    best_metric_IRR = None
    best_metric_MRR = None
    best_results_IRR = None
    best_results_MRR = None
    global_best_IRR = 999
    global_best_MRR = 0

    r_loss = torch.tensor([]).float().to(device)
    c_loss = torch.tensor([]).float().to(device)
    ra_loss = torch.tensor([]).float().to(device)

    for epoch in range(epochs):
        for week in range(num_weeks):
            model.train()  # Prep to train model
            batch_x1, batch_x2, batch_x3, batch_x4 = (
                train_w1[week].to(device),
                train_w2[week].to(device),
                train_w3[week].to(device),
                train_w4[week].to(device),
            )
            batch_weekly = [batch_x1, batch_x2, batch_x3, batch_x4][-agg_week_num:]
            batch_reg_y = train_reg[week].view(-1, 1).to(device)
            batch_cls_y = train_cls[week].view(-1, 1).to(device)
            reg_out, cls_out = model(batch_weekly)
            reg_out, cls_out = reg_out.view(-1, 1), cls_out.view(-1, 1)

            # Calculate loss
            reg_loss = reg_loss_func(reg_out, batch_reg_y)  # (target_size, 1)
            cls_loss = cls_loss_func(cls_out, batch_cls_y)
            rank_loss = torch.relu(-(reg_out.view(-1, 1) * reg_out.view(1, -1)) * (batch_reg_y.view(-1, 1) * batch_reg_y.view(1, -1)))
            c_loss = torch.cat((c_loss, cls_loss.view(-1, 1)))
            r_loss = torch.cat((r_loss, reg_loss.view(-1, 1)))
            ra_loss = torch.cat((ra_loss, rank_loss.view(-1, 1)))

            if (week+1) % 1 ==0:
                cls_loss = beta*torch.mean(c_loss)
                reg_loss = alpha*torch.mean(r_loss)
                rank_loss = gamma*torch.sum(ra_loss)
                loss = reg_loss + rank_loss + cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                r_loss = torch.tensor([]).float().to(device)
                c_loss = torch.tensor([]).float().to(device)
                ra_loss = torch.tensor([]).float().to(device)
                if (week+1) % 144 ==0:  
                    print("REG Loss:%.4f CLS Loss:%.4f RANK Loss:%.4f  Loss:%.4f"% (reg_loss.item(),cls_loss.item(),rank_loss.item(),loss.item()))

        
        # Evaluate
        model.eval()
        print("Evaluate at epoch %s" % (epoch + 1))
        print("-----------------------------------------------")
        y_pred, y_pred_cls = model.predict_toprank([test_w1, test_w2, test_w3, test_w4], device, top_k=5)

        # Calculate metric
        y_pred = np.array(y_pred).ravel()
        test_y = np.array(test_y).ravel()
        mae = round(mean_absolute_error(test_y, y_pred[:18050]), 4)
        acc_score = Acc(test_cls, y_pred)

        print("Accuracy at epoch ", {epoch+1}, ": ", acc_score)
        print("-----------------------------------------------")
        predictions = []
        y_pred = y_pred[:18050]
        predictions = y_pred

        results = []
        for k in ks_list:
            IRRs , MRRs ,Prs =[],[],[]
            for i in range(test_shape):
                M = MRR(np.array(test_y[loop_number*i:loop_number*(i+1)]),np.array(y_pred[loop_number*i:loop_number*(i+1)]),k=k)
                MRRs.append(M)
                P = Precision(np.array(test_y[loop_number*i:loop_number*(i+1)]),np.array(y_pred[loop_number*i:loop_number*(i+1)]),k=k)
                Prs.append(P)
            over_all = [mae,round(acc_score,4),round(np.mean(MRRs),4),round(np.mean(Prs),4)]
            results.append(over_all)


        performance = [round(mae, 4), round(acc_score, 4), round(np.mean(MRRs), 4), round(np.mean(Prs), 4)]

        if np.mean(MRRs) > global_best_MRR:
            global_best_MRR = np.mean(MRRs)
            best_metric_MRR = performance
            best_results_MRR =  results
            
    print("Finished Training!")
    print("-----------------------------------------------")
    torch.save(model.state_dict(), os.path.join("D:\\capstone_project\\capstone_project", "CAT" + "_model.pth"))
    top_stocks(predictions)
    print("-----------------------------------------------")
    return best_metric_IRR, best_metric_MRR, best_results_IRR, best_results_MRR

if __name__ == "__main__":    
    best_metric_IRR, best_metric_MRR, best_results_IRR, best_results_MRR = train(args)
    print()
    print("-------Final result-------")
    print("[BEST MRR] MAE:%.4f ACC:%.4f MRR:%.4f Precision:%.4f" % tuple(best_metric_MRR))
    for idx, k in enumerate(ks_list):
        print("[BEST RESULT MRR with k=%s] MAE:%.4f ACC:%.4f MRR:%.4f Precision:%.4f" % tuple(tuple([k])+tuple(best_results_MRR[idx])))