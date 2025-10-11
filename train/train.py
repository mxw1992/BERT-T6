import argparse
import torch
import os
from torch import nn
from transformers import BertModel,BertTokenizer, get_linear_schedule_with_warmup
import re
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
import torch.optim as optim
#import torch_optimizer as optim
import numpy as np
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                            f1_score, roc_auc_score, matthews_corrcoef, 
                            average_precision_score, confusion_matrix)

from model_def_2 import BERT_Classifier
import random

d_model = 1024
batch_size = 8
MAX_LEN = 1981
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

def alpha_cal(train_set_csv_path):
#    train_set_csv_path = './train_set.csv'
    training_data = pd.read_csv(train_set_csv_path)
    y = training_data.loc[:,"Label"]
    pos = 0
    neg =0 
    for i in y:
        pos += (i == 1)
        neg += (i == 0)
    return neg / (pos + neg)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0001, path='checkpoint.pt'):
        """
        Args:
            patience (int): 验证集性能不再提升后等待的epoch数
            verbose (bool): 是否打印早停信息
            delta (float): 认为有提升的最小变化量
            path (str): 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss  
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''保存模型当验证集损失减少时'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): 类别平衡权重（用于处理类别不平衡），默认为0.25。
                           通常设置为类别频率的倒数或通过交叉验证调优。
            gamma (float): 聚焦参数（gamma > 0），抑制易分类样本的损失贡献。
                          值越大，对难样本的关注越高。
            reduction (str): 损失聚合方式，可选 'mean'、'sum' 或 'none'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算标准交叉熵损失（未加权）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape: [batch_size]
        
        # 获取模型对真实类别的预测概率
        p_t = torch.exp(-ce_loss)  # p_t = exp(-CE), 等价于 softmax中对应类别的概率
        
        # Focal Loss 核心公式
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        # 根据 reduction 参数聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Seq_Dataset(Dataset):
    def __init__(self, sequence, targets, tokenizer, max_len):
        self.sequence = sequence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'protein_sequence': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


def _get_train_data_loader(batch_size, train_dir):
    dataset = pd.read_csv(train_dir)
    train_data = Seq_Dataset(
        sequence=dataset.SEQUENCE_space.to_numpy(),
        targets=dataset.Label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return train_dataloader

def _get_test_data_loader(batch_size, test_dir):
    dataset = pd.read_csv(test_dir)
    test_data = Seq_Dataset(
        sequence=dataset.SEQUENCE_space.to_numpy(),
        targets=dataset.Label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_dataloader


def freeze(model, frozen_layers):
    modules = [model.bert.encoder.layer[:frozen_layers]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir)
    test_loader = _get_train_data_loader(1, args.valid_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # set the seed for generating random numbers

        # 初始化早停
    early_stopping = EarlyStopping(
        patience=args.patience, 
        verbose=True, 
        path=args.checkpoint_path
    )

    model = BERT_Classifier()
    freeze(model, args.frozen_layers)
    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda x: x.requires_grad is not False,
        model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)



    alpha = alpha_cal(args.train_dir)
    loss_fn = FocalLoss(alpha = alpha)
    #loss_fn = nn.CrossEntropyLoss()

    
    test_max_ACC = 0.5
    test_max_F1_epoch = 1
    test_max_Sn = 0.5
    test_max_Sp = 0.5
    test_max_Precision = 0.5
    test_max_F1 = 0.0
    test_max_AUC_ROC = 0.5
    test_max_MCC = 0.0
    test_max_AUC_PRC = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        y_true, y_pred, y_probs = [], [], []
        for batch in train_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)
            
            output = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(output, b_labels)
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            optimizer.zero_grad()

            probs = torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy()  # 正类概率
            preds = (probs > 0.5).astype(int)
            
            # y_true.extend(torch.max(target, dim=1))
            y_true.extend(b_labels.cpu())
            y_pred.extend(preds)
            y_probs.extend(probs)
            


        y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)
        
            
    
    # 计算所有指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        train_metrics = {
              'ACC': accuracy_score(y_true, y_pred),
              'Sn': tp / (tp + fn),  # Sensitivity/Recall
              'Sp': tn / (tn + fp),  # Specificity
              'Precision': precision_score(y_true, y_pred, zero_division=0),
              'F1': f1_score(y_true, y_pred),
              'AUC_ROC': roc_auc_score(y_true, y_probs),
              'MCC': matthews_corrcoef(y_true, y_pred),
              'AUC_PRC': average_precision_score(y_true, y_probs)
             }

    
        test_y_true, test_y_pred, test_y_probs = [], [], []
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['targets'].to(device)
                
                output = model(b_input_ids, attention_mask=b_input_mask)
                valid_loss = loss_fn(output, b_labels)
                test_losses.append(valid_loss.item())
                
                probs = torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy()  # 正类概率
                preds = (probs > 0.5).astype(int)

                test_y_pred.extend(preds)
                test_y_true.extend(b_labels.cpu())
                test_y_probs.extend(probs)
        
        # 计算验证集平均损失
        avg_test_loss = np.mean(test_losses)
        
        # 早停检查
        early_stopping(avg_test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        test_y_true, test_y_pred, test_y_probs = np.array(test_y_true), np.array(test_y_pred), np.array(test_y_probs)
        
        
        tn, fp, fn, tp = confusion_matrix(test_y_true, test_y_pred).ravel()
        test_metrics = {
              'ACC': accuracy_score(test_y_true, test_y_pred),
              'Sn': tp / (tp + fn),  # Sensitivity/Recall
              'Sp': tn / (tn + fp),  # Specificity
              'Precision': precision_score(test_y_true, test_y_pred, zero_division=0),
              'F1': f1_score(test_y_true, test_y_pred),
              'AUC_ROC': roc_auc_score(test_y_true, test_y_probs),
              'MCC': matthews_corrcoef(test_y_true, test_y_pred),
              'AUC_PRC': average_precision_score(test_y_true, test_y_probs)
             }


        if test_metrics['F1'] > test_max_F1:
            test_max_ACC = test_metrics['ACC']
            test_max_F1_epoch = epoch
            test_max_Sn = test_metrics['Sn']
            test_max_Sp = test_metrics['Sp']
            test_max_Precision = test_metrics['Precision']
            test_max_F1 = test_metrics['F1']
            test_max_AUC_ROC = test_metrics['AUC_ROC']
            test_max_MCC = test_metrics['MCC']
            test_max_AUC_PRC = test_metrics['AUC_PRC']
        print("epoch:", epoch, 'train_ACC =', '{:.4f}'.format(train_metrics['ACC']), 'train_Sn =', '{:.4f}'.format(train_metrics['Sn']),
              'train_Sp =', '{:.4f}'.format(train_metrics['Sp']),'train_Precision =', '{:.4f}'.format(train_metrics['Precision']),
              'train_F1 =', '{:.4f}'.format(train_metrics['F1']),'train_AUC_ROC =', '{:.4f}'.format(train_metrics['AUC_ROC']),
               'train_MCC =', '{:.4f}'.format(train_metrics['MCC']),'train_AUC_PRC =', '{:.4f}'.format(train_metrics['AUC_PRC']),
               'test_ACC =', '{:.4f}'.format(test_metrics['ACC']), 'test_Sn =', '{:.4f}'.format(test_metrics['Sn']),
              'test_Sp =', '{:.4f}'.format(test_metrics['Sp']),'test_Precision =', '{:.4f}'.format(test_metrics['Precision']),
               'test_F1 =', '{:.4f}'.format(test_metrics['F1']),'test_AUC_ROC =', '{:.4f}'.format(test_metrics['AUC_ROC']),
               'test_MCC =', '{:.4f}'.format(test_metrics['MCC']),'test_AUC_PRC =', '{:.4f}'.format(test_metrics['AUC_PRC']),
              )
        result_dict = {"seed": args.seed, "epochs": epoch, "train_ACC": train_metrics['ACC'], "train_Sn": train_metrics['Sn'], "train_Sp": train_metrics['Sp'], 
    	               "train_Precision": train_metrics['Precision'], "train_F1": train_metrics['F1'], "train_AUC_ROC": train_metrics['AUC_ROC'], 
    	    	       "train_MCC": train_metrics['MCC'], "train_AUC_PRC": train_metrics['AUC_PRC'],
              "test_ACC": test_metrics['ACC'], "test_Sn": test_metrics['Sn'], "test_Sp": test_metrics['Sp'], "test_Precision": test_metrics['Precision'],
               "test_F1": test_metrics['F1'], "test_AUC_ROC": test_metrics['AUC_ROC'], "test_MCC": test_metrics['MCC'], "test_AUC_PRC": test_metrics['AUC_PRC'],
    	           "test_max_F1_epoch":test_max_F1_epoch, "best_ACC": test_max_ACC, "best_Sn": test_max_Sn,"best_Sp": test_max_Sp, "best_Precision": test_max_Precision, 
                "best_F1": test_max_F1, "best_AUC_ROC": test_max_AUC_ROC, "best_MCC": test_max_MCC, "best_AUC_PRC": test_max_AUC_PRC}
        result_df = pd.DataFrame(result_dict,index=[epoch])
        result_df.columns = ['seed','epoch','train_ACC','train_Sn','train_Sp','train_Precision','train_F1','train_AUC_ROC','train_MCC','train_AUC_PRC',
        'test_ACC','test_Sn','test_Sp','test_Precision','test_F1','test_AUC_ROC','test_MCC','test_AUC_PRC','test_max_F1_epoch','best_ACC','best_Sn','best_Sp',
        'best_Precision','best_F1','best_AUC_ROC','best_MCC','best_AUC_PRC']

        result_df.to_csv(args.result_dir, mode='a', header=not os.path.exists(args.result_dir))
    return test_max_ACC, test_max_Sn, test_max_Sp, test_max_Precision, test_max_F1,test_max_AUC_ROC,test_max_MCC, test_max_AUC_PRC

if __name__ == "__main__":

    train_path = "./train_set.csv"
    valid_path = "./valid_set.csv"

    my_seed = 42
    result_path = "./training_results.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=my_seed, metavar="S", help="random seed (default: 42)")
    parser.add_argument("--train_dir", type=str, default=train_path)
    parser.add_argument("--valid_dir", type=str, default=valid_path)
    parser.add_argument("--result_dir", type=str, default=result_path)
    parser.add_argument(
     "--batch-size", type=int, default=6, metavar="N", help="input batch size for training (default: 6)"
    )

    parser.add_argument("--frozen_layers", type=int, default=0, metavar="NL",
                          help="number of frozen layers(default: 10)")
    parser.add_argument("--lr", type=float, default=1e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
    parser.add_argument("--weight_decay", type=float, default=5e-3, metavar="M",
                          help="weight_decay (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                          help="number of epochs to train (default: 100)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoint_path", type=str, default="./best_model.pt", 
                       help="Path to save the best model")
    best_test_ACC, best_test_Sn,best_test_Sp,best_test_Precision,best_test_F1,best_test_AUC_ROC,best_test_MCC,best_test_AUC_PRC = train(parser.parse_args())

    print("smart")
