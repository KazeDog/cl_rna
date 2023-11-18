import math
import torch
from torch import nn
from torch.nn import functional as F
from layers import MLP, fc_layer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1001):  # ninp, dropout
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 5000 * 200
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 5000 * 1 * 200, 最长5000的序列，每个词由1 * 200的矩阵代表着不同的时间
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x)
        x = x + self.pe[:x.size()[0], :]        # torch.Size([35, 1, 200])
        return self.dropout(x)

class Classifier(nn.Module):

    def __init__(self,  classes, seqlen,
                 # -fc-layers
                 fc_layers=3, fc_units=400, fc_drop=0, fc_bn=False, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=True, phantom=False):
        super().__init__()

        self.param_list = [self.named_parameters]
        self.optimizer = None
        self.optim_type = "adam"
        # --> self.[optim_type]   <str> name of optimizer, relevant if optimizer should be reset for every context
        self.optim_list = []
        # --> self.[optim_list]   <list>, if optimizer should be reset after each context, provide list of required <dicts>

        # Atributes defining how to use memory-buffer
        self.prototypes = False  # -> perform classification by using prototypes as class templates
        self.add_buffer = False  # -> add the memory buffer to the training set of the current task

        self.classes = classes
        self.label = "Classifier"
        self.hidden_dim = 64
        self.batch_size = 32
        self.emb_dim = 128
        self.seq_len = seqlen

        self.binaryCE = False
        self.binaryCE_distill = False

        self.embedding_RNA = nn.Embedding(5, self.emb_dim)

        self.position_encode_RNA = PositionalEncoding(d_model=self.emb_dim, max_len=1001)

        self.encoder_layer_RNA = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=4)

        # Pass the sequence information and secondary structure information through different transformer encoders
        self.transformer_encoder_RNA = nn.TransformerEncoder(self.encoder_layer_RNA, num_layers=1)

        self.gru_RNA = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                             bidirectional=True, dropout=0.2)

        self.block_RNA = nn.Sequential(nn.Linear(self.hidden_dim * 2 * self.seq_len + 2 * self.hidden_dim * 2, 4096),
                                       nn.LeakyReLU(),
                                       nn.Linear(4096, 1024))

        self.fcE = MLP(input_size=self.hidden_dim * 2 * self.seq_len + 2 * self.hidden_dim * 2, output_size=fc_units, layers=fc_layers-1,
                       hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated, phantom=phantom)

        mlp_output_size = fc_units if fc_layers > 1 else self.conv_out_units
        self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop,
                                   phantom=phantom)
        self.fcE.frozen = False

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @property
    def name(self):
        return 'classify'

    def feature_extractor(self, images):
        x = self.embedding_RNA(images)
        x = self.position_encode_RNA(x.permute(1, 0, 2))  # [243, 128, 128]
        x_output = self.transformer_encoder_RNA(x)  # [243, 128, 128]
        x_output, x_hn = self.gru_RNA(x_output)  # [243, 128, 50]
        x_output = x_output.permute(1, 0, 2)
        x_hn = x_hn.permute(1, 0, 2)
        x_output = x_output.reshape(x_output.shape[0], -1)  # [128, 12150]
        x_hn = x_hn.reshape(x_output.shape[0], -1)
        x_output = torch.cat([x_output, x_hn], 1)  # [128, 12250]

        x_output = self.fcE(x_output)  # [128, 1024]

        representation = x_output  # [128, 1024]
        pred = self.classifier(representation)  # [128, 2]

        return pred

    def classify(self, x, allowed_classes=None, no_prototypes=False):
        '''For input [x] (image/"intermediate" features), return predicted "scores"/"logits" for [allowed_classes].'''
        if self.prototypes and not no_prototypes:
            return self.classify_with_prototypes(x, allowed_classes=allowed_classes)
        else:
            scores = self.forward(x)
            return scores if (allowed_classes is None) else scores[:, allowed_classes]
        # return self.forward(x)

    def get_map(self, x):
        self.gru_RNA.flatten_parameters()
        x = self.embedding_RNA(x)
        x = self.position_encode_RNA(x.permute(1, 0, 2))  # [243, 128, 128]
        x_output = self.transformer_encoder_RNA(x)  # [243, 128, 128]
        x_output, x_hn = self.gru_RNA(x_output)  # [243, 128, 50]
        x_output = x_output.permute(1, 0, 2)
        x_hn = x_hn.permute(1, 0, 2)
        x_output = x_output.reshape(x_output.shape[0], -1)  # [128, 12150]
        x_hn = x_hn.reshape(x_output.shape[0], -1)
        x_output = torch.cat([x_output, x_hn], 1)  # [128, 12250]

        return x_output


    def forward(self, x):
        x = self.embedding_RNA(x)
        x = self.position_encode_RNA(x.permute(1, 0, 2))  # [243, 128, 128]
        x_output = self.transformer_encoder_RNA(x)  # [243, 128, 128]
        x_output, x_hn = self.gru_RNA(x_output)  # [243, 128, 50]
        x_output = x_output.permute(1, 0, 2)
        x_hn = x_hn.permute(1, 0, 2)
        x_output = x_output.reshape(x_output.shape[0], -1)  # [128, 12150]
        x_hn = x_hn.reshape(x_output.shape[0], -1)
        x_output = torch.cat([x_output, x_hn], 1)  # [128, 12250]
        x_output = self.fcE(x_output)  # [128, 1024]

        representation = x_output  # [128, 1024]
        pred = self.classifier(representation)  # [128, 2]

        return pred

