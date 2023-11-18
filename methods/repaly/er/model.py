import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from layers import MLP, fc_layer


def to_one_hot(y, classes, device=None):
    '''Convert <nd-array> or <tensor> with integers [y] to a 2D "one-hot" <tensor>.'''
    if type(y)==torch.Tensor:
        device=y.device
        y = y.cpu()
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c if device is None else c.to(device)

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
                 fc_layers=3, fc_units=256, fc_drop=0, fc_bn=False, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=True, phantom=False, epsilon=0.1):
        super().__init__()

        # List with the methods_old to create generators that return the parameters on which to apply param regularization
        self.param_list = [self.named_parameters]  # -> lists the parameters to regularize with SI or diagonal Fisher
        #   (default is to apply it to all parameters of the network)

        # Optimizer (and whether it needs to be reset)
        self.optimizer = None
        self.optim_type = "adam"
        # --> self.[optim_type]   <str> name of optimizer, relevant if optimizer should be reset for every context
        self.optim_list = []
        # --> self.[optim_list]   <list>, if optimizer should be reset after each context, provide list of required <dicts>

        # Scenario, singlehead & negative samples
        self.scenario = 'task'  # which scenario will the model be trained on
        self.classes_per_context = 2  # number of classes per context
        self.singlehead = False  # if Task-IL, does the model have a single-headed output layer?
        self.neg_samples = 'all'  # if Class-IL, which output units should be set to 'active'?

        self.mask_dict = None

        self.replay_mode = "none"    # should replay be used, and if so what kind? (none|current|buffer|all|generative)
        self.replay_targets = "hard" # should distillation loss be used? (hard|soft)
        self.KD_temp = 2.            # temperature for distillation loss
        self.use_replay = "normal"   # how to use the replayed data? (normal|inequality|both)
                                     # -inequality = use gradient of replayed data as inequality constraint for gradient
                                     #               of the current data (as in A-GEM; Chaudry et al., 2019; ICLR)
        self.eps_agem = 0.           # parameter that improves numerical stability of AGEM (if set slighly above 0)
        self.lwf_weighting = False   # LwF has different weighting of the 'stability' and 'plasticity' terms than replay

        self.classes = classes
        self.label = "Classifier"
        self.hidden_dim = 64
        self.batch_size = 32
        self.emb_dim = 128
        self.seq_len = seqlen

        # settings for training
        self.binaryCE = False  # -> use binary (instead of multiclass) prediction error
        self.binaryCE_distill = False  # -> for classes from previous contexts, use the by the previous model
        #   predicted probs as binary targets (only in Class-IL with binaryCE)

        # Replay
        # List with memory-sets
        self.memory_sets = []  # -> each entry of [self.memory_sets] is an <np.array> of N images with shape (N, Ch, H, W)
        # self.memory_sets = {}
        self.memory_set_means = []
        self.compute_means = True

        # Settings
        self.use_memory_buffer = True
        self.budget = 0.1
        self.use_full_capacity = False
        self.sample_selection = 'random'
        self.norm_exemplars = True

        # Atributes defining how to use memory-buffer
        self.prototypes = False  # -> perform classification by using prototypes as class templates
        self.add_buffer = False  # -> add the memory buffer to the training set of the current task

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
        self.classifier = fc_layer(mlp_output_size, self.classes, excit_buffer=True, nl='none', drop=fc_drop,
                                   phantom=phantom)
        self.fcE.frozen = False

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    ####----MANAGING THE MEMORY BUFFER----####

    def reduce_memory_sets(self, m):
        for y, P_y in enumerate(self.memory_sets):
            self.memory_sets[y] = P_y[:m]


    def construct_memory_set(self, dataset, n):
        '''Construct memory set of [n] examples from [dataset] using 'herding', 'random' or 'fromp' selection.

        Note that [dataset] should be from specific class; selected sets are added to [self.memory_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        memory_set = []

        indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
        for k in indeces_selected:
            memory_set.append(dataset[k][0].numpy())
        self.memory_sets.append(np.array(memory_set))

        self.train(mode=mode)


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
        scores = self.forward(x)
        return scores if (allowed_classes is None) else scores[:, allowed_classes]

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
        x_output = self.fcE(x_output)  # [128, 1024]

        representation = x_output  # [128, 1024]
        pred = self.classifier(representation)  # [128, 2]

        return pred
