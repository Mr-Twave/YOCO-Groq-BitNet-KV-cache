import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class SelfDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(SelfDecoder, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(hidden_size, 8) for _ in range(num_layers)])
        
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x

class CrossDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(CrossDecoder, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(hidden_size, 8) for _ in range(num_layers)])
        
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x

class YOCOModel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(YOCOModel, self).__init__()
        self.self_decoder = SelfDecoder(hidden_size, num_layers)
        self.cross_decoder = CrossDecoder(hidden_size, num_layers)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # BitNet integration
        self.bitnet = BitNet(hidden_size)
        
    def forward(self, input_ids, attention_mask=None):
        # Encode the input using BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        memory = outputs.last_hidden_state
        
        # Apply BitNet
        bit_memory = self.bitnet(memory)
        
        # Self-decoder
        self_decoded = self.self_decoder(bit_memory, bit_memory)
        
        # Cross-decoder
        cross_decoded = self.cross_decoder(self_decoded, bit_memory)
        
        return cross_decoded

class BitNet(nn.Module):
    def __init__(self, hidden_size):
        super(BitNet, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        return self.linear(x.sign())
