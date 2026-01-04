from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertForMaskedLM, BertTokenizer, pipeline
import torch
import torch.nn.functional as F
import torch.nn as nn

BertModel.from_pretrained("Rostlab/prot_bert")
class BERT_Classifier(nn.Module):
    def __init__(self):
        super(BERT_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert",output_attentions=True)
        self.Classifier= nn.Sequential(nn.LayerNorm(self.bert.config.hidden_size),
                                      #
                                      nn.Linear(self.bert.config.hidden_size, 512),
                                      nn.LeakyReLU(inplace=False),
                                      nn.Dropout(p=0.4),
                                      nn.Linear(512, 2))

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        attentions = output.attentions
        last_layer_attention = attentions[-1]  # 最后一层attention [batch_size, num_heads, seq_len, seq_len]
        return self.Classifier(output.pooler_output),last_layer_attention
