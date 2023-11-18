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

        self.weight_penalty = True
        self.importance_weighting = 'si'
        self.si_c = 10. if self.scenario == 'task' else (50000. if self.scenario == 'domain' else 5000.)
        self.reg_strength = self.si_c
        self.epsilon = epsilon

        self.prototypes = False

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

    # ------------- "Synaptic Intelligence"-specifc functions -------------#

    def register_starting_param_values(self):
        '''Register the starting parameter values into the model as a buffer.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_SI_prev_context'.format(n), p.detach().clone())

    def prepare_importance_estimates_dicts(self):
        '''Prepare <dicts> to store running importance estimates and param-values before update.'''
        W = {}
        p_old = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()
        return W, p_old

    def update_importance_estimates(self, W, p_old):
        '''Update the running parameter importance estimates in W.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad * (p.detach() - p_old[n]))
                    p_old[n] = p.detach().clone()

    def update_omega(self, W, epsilon):
        '''After completing training on a context, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed context
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')

                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self, '{}_SI_prev_context'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W[n] / (p_change ** 2 + epsilon)
                    try:
                        omega = getattr(self, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add

                    # Store these new values in the model
                    self.register_buffer('{}_SI_prev_context'.format(n), p_current)
                    self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for gen_params in self.param_list:
                for n, p in gen_params():
                    # print('p.requires_grad=', p.requires_grad)
                    if p.requires_grad:
                        # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                        n = n.replace('.', '__')
                        prev_values = getattr(self, '{}_SI_prev_context'.format(n))
                        omega = getattr(self, '{}_SI_omega'.format(n))
                        # Calculate SI's surrogate loss, sum over all parameters
                        losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

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
