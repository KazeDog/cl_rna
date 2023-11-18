import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np

class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''
    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr

def linearExcitability(input, weight, excitability=None, bias=None):
    '''Applies a linear transformation to the incoming data: :math:`y = c(xA^T) + b`.

    Shape:
        - input:        :math:`(N, *, in_features)`
        - weight:       :math:`(out_features, in_features)`
        - excitability: :math:`(out_features)`
        - bias:         :math:`(out_features)`
        - output:       :math:`(N, *, out_features)`
    (NOTE: `*` means any number of additional dimensions)'''

    # print('input=', input)
    # print('input.size=', input.size())
    # print('weight=', weight)
    # print('weight.size=', weight.size())
    if excitability is not None:
        # print('input size=', input.size())
        # print('excitability size=', excitability.size())
        output = input.matmul(weight.t()) * excitability
    else:
        output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


class LinearExcitability(nn.Module):
    '''Module for a linear transformation with multiplicative excitability-parameter (i.e., learnable) and/or -buffer.

    Args:
        in_features:    size of each input sample
        out_features:   size of each output sample
        bias:           if 'False', layer will not learn an additive bias-parameter (DEFAULT=True)
        excitability:   if 'True', layer will learn a multiplicative excitability-parameter (DEFAULT=False)
        excit_buffer:   if 'True', layer will have excitability-buffer whose value can be set (DEFAULT=False)

    Shape:
        - input:    :math:`(N, *, in_features)` where `*` means any number of additional dimensions
        - output:   :math:`(N, *, out_features)` where all but the last dimension are the same shape as the input.

    Attributes:
        weight:         the learnable weights of the module of shape (out_features x in_features)
        excitability:   the learnable multiplication terms (out_features)
        bias:           the learnable bias of the module of shape (out_features)
        excit_buffer:   fixed multiplication variable (out_features)'''

    def __init__(self, in_features, out_features, bias=True, excitability=False, excit_buffer=False):
        super(LinearExcitability, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if excitability:
            self.excitability = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('excitability', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if excit_buffer:
            buffer = torch.Tensor(out_features).uniform_(1,1)
            self.register_buffer("excit_buffer", buffer)
        else:
            self.register_buffer("excit_buffer", None)
        self.reset_parameters()

    def reset_parameters(self):
        '''Modifies the parameters "in-place" to initialize / reset them at appropriate values.'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.excitability is not None:
            self.excitability.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        '''Running this model's forward step requires/returns:
            -[input]:   [batch_size]x[...]x[in_features]
            -[output]:  [batch_size]x[...]x[hidden_features]'''
        if self.excit_buffer is None:
            excitability = self.excitability
        elif self.excitability is None:
            excitability = self.excit_buffer
        else:
            excitability = self.excitability*self.excit_buffer
        return linearExcitability(input, self.weight, excitability, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'

class fc_layer(nn.Module):
    '''Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size, nl=nn.ReLU(), drop=0., bias=True, batch_norm=False,
                 excitability=False, excit_buffer=False, gated=False, phantom=False):
        super().__init__()
        self.bias = False if batch_norm else bias
        if drop>0:
            self.dropout = nn.Dropout(drop)
        self.linear = LinearExcitability(in_size, out_size, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gated:
            self.gate = nn.Linear(in_size, out_size)
            self.sigmoid = nn.Sigmoid()
        if phantom:
            self.phantom = nn.Parameter(torch.zeros(out_size), requires_grad=True)
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl=="none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else Identity())

    def forward(self, x, return_pa=False, **kwargs):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        # print('input.size=', input.size())
        # print('hasattr(self, bn)=', hasattr(self, 'bn'))
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        if hasattr(self, 'phantom'):
            gated_pre_activ = gated_pre_activ + self.phantom
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]

class fc_layer_split(nn.Module):
    '''Fully connected layer outputting [mean] and [logvar] for each unit.

    Input:  [batch_size] x ... x [in_size] tensor
    Output: tuple with two [batch_size] x ... x [out_size] tensors'''

    def __init__(self, in_size, out_size, nl_mean=nn.Sigmoid(), nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False, gated=False):
        super().__init__()

        self.mean = fc_layer(in_size, out_size, drop=drop, bias=bias, excitability=excitability,
                             excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_mean)
        self.logvar = fc_layer(in_size, out_size, drop=drop, bias=False, excitability=excitability,
                               excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_logvar)

    def forward(self, x):
        return (self.mean(x), self.logvar(x))

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list

class fc_layer_fixed_gates(nn.Module):
    '''Fully connected layer, with possibility of returning "pre-activations". Has fixed gates (of specified dimension).

    Input:  [batch_size] x ... x [in_size] tensor         &        [batch_size] x ... x [gate_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size, nl=nn.ReLU(),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False,
                 gate_size=0, gating_prop=0.8, device='cpu'):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = LinearExcitability(in_size, out_size, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gate_size>0:
            self.gate_mask = torch.tensor(
                np.random.choice([0., 1.], size=(gate_size, out_size), p=[gating_prop, 1.-gating_prop]),
                dtype=torch.float, device=device
            )
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == "none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else Identity())

    def forward(self, x, gate_input=None, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = torch.matmul(gate_input, self.gate_mask) if hasattr(self, 'gate_mask') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate_mask') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]

class MLP(nn.Module):
    '''Module for a multi-layer perceptron (MLP).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor'''

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None,
                 drop=0, batch_norm=False, nl="relu", bias=True, excitability=False, excit_buffer=False, gated=False,
                 phantom=False, output='normal'):
        '''sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gated]            <bool>; if True, each linear layer has an additional learnable gate
                                    (whereby the gate is controlled by the same input as that goes through the gate)
        [phantom]          <bool>; if True, add phantom parameters to pre-activations, used for computing KFAC Fisher
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity'''

        super().__init__()
        self.output = output

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if (hid_smooth is not None):
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers-1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size]
        print('size_per_layer=', size_per_layer)
        self.layers = len(size_per_layer)-1

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}{gate}".format(
            drop="" if drop==0 else "d{}".format(drop),
            bias="" if bias else "n", exc="e" if excitability else "", bn="b" if batch_norm else "",
            nl="l" if nl=="leakyrelu" else ("n" if nl=="none" else ""), gate="g" if gated else "",
        )
        nd_label = "{}{}".format("" if nd_label=="" else "-{}".format(nd_label),
                                 "" if output=="normal" else "-{}".format(output))
        # -set label
        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement=="" else "x", i)
        print('size_statement=', size_statement)
        self.label = "F{}{}".format(size_statement, nd_label) if self.layers>0 else ""
        print('label=', self.label)

        # set layers
        for lay_id in range(1, self.layers+1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id-1]
            out_size = size_per_layer[lay_id]
            # define and set the fully connected layer
            layer = fc_layer(
                in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer,
                batch_norm=False if (lay_id==self.layers and not output=="normal") else batch_norm, gated=gated,
                nl=("none" if output=="none" else nn.Sigmoid()) if (
                    lay_id==self.layers and not output=="normal"
                ) else nl, drop=drop if lay_id>1 else 0., phantom=phantom
            )
            setattr(self, 'fcLayer{}'.format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers<1:
            self.noLayers = Identity()

    def forward(self, x, return_intermediate=False):
        if return_intermediate:
            intermediate = {}
        for lay_id in range(1, self.layers + 1):
            if return_intermediate:
                intermediate[f"fcLayer{lay_id}"] = x
            x = getattr(self, "fcLayer{}".format(lay_id))(x)
        return (x, intermediate) if return_intermediate else x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.layers+1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list

class MLP_gates(nn.Module):
    '''Module for a multi-layer perceptron (MLP). Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.
    With gates controlled by [gate_input] (of size [gate_size]) with a randomly selected masked (prop=[gating_prop]).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor         &        [batch_size] x [gate_size]
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor'''

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None,
                 drop=0, batch_norm=False, nl="relu", bias=True, excitability=False, excit_buffer=False, gate_size=0,
                 gating_prop=0., final_gate=False, output='normal', device='cpu'):
        '''sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gate_size]        <int>; if>0, each linear layer has gate controlled by separate inputs of size [gate_size]
        [gating_prop]      <float>; probability for each unit to be gated
        [final_gate]       <bool>; whether final layer is allowed to have a gate
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity'''

        super().__init__()
        self.output = output

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if (hid_smooth is not None):
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers-1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size] if layers>0 else [input_size]
        self.layers = len(size_per_layer)-1

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}{gate}".format(
            drop="" if drop==0 else "d{}".format(drop),
            bias="" if bias else "n", exc="e" if excitability else "", bn="b" if batch_norm else "",
            nl="l" if nl=="leakyrelu" else ("n" if nl=="none" else ""),
            gate="g{}m{}".format(gate_size, gating_prop) if (gate_size>0 and gating_prop>0.) else "",
        )
        nd_label = "{}{}".format("" if nd_label=="" else "-{}".format(nd_label),
                                 "" if output=="normal" else "-{}".format(output))
        # -set label
        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement=="" else "x", i)
        self.label = "F{}{}".format(size_statement, nd_label) if self.layers>0 else ""

        # set layers
        for lay_id in range(1, self.layers+1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id-1]
            out_size = size_per_layer[lay_id]
            # define and set the fully connected layer
            if (not gate_size>0.) or (not gating_prop>0.) or (lay_id==self.layers and not final_gate):
                layer = fc_layer(
                    in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer,
                    batch_norm=False if (lay_id==self.layers and not output=="normal") else batch_norm,
                    nl=("none" if output=="none" else nn.Sigmoid()) if (
                        lay_id==self.layers and not output=="normal"
                    ) else nl, drop=drop if lay_id>1 else 0.,
                )
            else:
                layer = fc_layer_fixed_gates(
                    in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer,
                    batch_norm=False if (lay_id == self.layers and not output == "normal") else batch_norm,
                    gate_size=gate_size, gating_prop=gating_prop, device=device,
                    nl=("none" if output == "none" else nn.Sigmoid()) if (
                            lay_id == self.layers and not output == "normal"
                    ) else nl, drop=drop if lay_id>1 else 0.,
                )
            setattr(self, 'fcLayer{}'.format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers<1:
            self.noLayers = Identity()

    def forward(self, x, gate_input=None, skip_first=0, skip_last=0, return_lists=False):
        # Initiate <list> for keeping track of intermediate hidden-(pre)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all fc-layers
        for lay_id in range(skip_first+1, self.layers+1-skip_last):
            (x, pre_act) = getattr(self, 'fcLayer{}'.format(lay_id))(x, gate_input=gate_input, return_pa=True)
            if return_lists:
                pre_act_list.append(pre_act)  #-> for each layer, store pre-activations
                if lay_id<(self.layers-skip_last):
                    hidden_act_list.append(x) #-> for all but last layer, store hidden activations
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x


    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.layers+1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list
