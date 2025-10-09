import argparse
from transformers import BertModel,BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from model_def_2 import BERT_Classifier
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                            f1_score, roc_auc_score, matthews_corrcoef, 
                            average_precision_score, confusion_matrix)

import fasta2csv

d_model = 1024
batch_size = 6
MAX_LEN = 1981
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

class Seq_Dataset(Dataset):
    def __init__(self, sequence, tokenizer, max_len):
        self.sequence = sequence
#        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
#        target = self.targets[item]
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
#          'targets': torch.tensor(target, dtype=torch.long)
        }

def _get_test_data_loader(batch_size, test_dir):
    dataset = pd.read_csv(test_dir)
    test_data = Seq_Dataset(
        sequence=dataset.SEQUENCE_space.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_dataloader


def freeze(model, frozen_layers):
    modules = [model.bert.encoder.layer[:frozen_layers]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def test(args):
    seq_name, csv_path = fasta2csv.fasta2csv(args.test_dir)
    test_loader = _get_test_data_loader(1, csv_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # set the seed for generating random numbers

    model = BERT_Classifier()
    model.load_state_dict(torch.load(args.checkpoint_path,map_location=device))
    model.to(device)

    freeze(model, args.frozen_layers)
   # model = model.to(device)
    
    test_y_pred, test_y_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
#            b_labels = batch['targets'].to(device)
            output = model(b_input_ids, attention_mask=b_input_mask)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy()  # 正类概率
            preds = (probs > 0.5).astype(int)

            test_y_pred.extend(preds)
            test_y_probs.extend(probs)
        
#    test_y_pred, test_y_probs = np.array(test_y_pred), np.array(test_y_probs)
    T6SE = []
    for i in test_y_pred:
        if i == 1:
    	        T6SE.append('Yes')
        else:
            T6SE.append('No')
        
    result_df = pd.DataFrame({     
                'ID': seq_name,     
                'Pred': test_y_probs,     
                'T6SE': T6SE})   

    result_df.to_csv(args.result_dir,index=False)
    return result_df

if __name__ == "__main__":

    test_path = "./data/prediction.csv"

    result_path = "./data/" + "/prediction_result.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default=test_path)
    parser.add_argument("--result_dir", type=str, default=result_path)
    parser.add_argument(
     "--batch-size", type=int, default=1, metavar="N", help="input batch size for predicting (default: 1)"
    )

    parser.add_argument("--frozen_layers", type=int, default=0, metavar="NL",
                          help="number of frozen layers(default: 0)")
    parser.add_argument("--checkpoint_path", type=str, default="./best_model.pt", 
                       help="Path to save the best model")

    results =  test(parser.parse_args())

    print("smart")
