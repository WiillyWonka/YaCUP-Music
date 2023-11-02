import torch
from torch import nn, Tensor
import torch.nn.functional as F
from base import BaseModel
from transformers import BertForSequenceClassification
from typing import List
import math

class BaselineModel(nn.Module):
    def __init__(
        self,
        num_classes = 256,
        input_dim = 768,
        hidden_dim = 512
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(hidden_dim)
        self.projector =  nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        

    def forward(self, embeds):
        x = [self.projector(x) for x in embeds]
        x = [v.mean(0).unsqueeze(0) for v in x]
        x = self.bn(torch.cat(x, dim = 0))
        x = self.lin(x)
        outs = self.fc(x)
        return outs

class BERT(nn.Module):
    def __init__(
        self,
        num_classes = 256
    ):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes, problem_type="multi_label_classification")

    def forward(self, embeds):
        out = self.model(inputs_embeds=embeds)
        return out.logits
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        d_model = 768
        nhead = 2
        num_layers = 1
        dropout = 0.1
        n_classes = 256
        dim_feedforward = 2048

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dim_feedforward=dim_feedforward)
        
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(d_model, 1)

        self.query_embed = nn.Embedding(n_classes, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)


    def forward(self, embeddings: List[Tensor]) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """

        out_batch = []

        for sample_emb in embeddings:
            sample_emb = sample_emb.unsqueeze(0)
            sample_emb = self.pos_encoder(sample_emb)
            output = self.transformer_decoder(self.query_embed.weight.unsqueeze(0), sample_emb)
            output = self.linear(output).view(1, -1)

            out_batch.append( output )

        out_batch = torch.cat(out_batch, dim=0)

        return out_batch