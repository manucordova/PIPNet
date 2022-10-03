###########################################################################
###                               PIPNet                                ###
###                          Model definition                           ###
###                        Author: Manuel Cordova                       ###
###       Adapted from https://github.com/ndrplz/ConvLSTM_pytorch       ###
###                       Last edited: 2022-09-23                       ###
###########################################################################

import numpy as np
import torch
import torch.nn as nn



def get_act(act):
    """
    Get an activation function from its name

    Input:  - act   Activation function name

    Output: - actf  Activation function
    """
    
    act = act.lower()

    if act == "linear":
        return nn.Identity()
    if act == "relu":
        return nn.ReLU()
    if act == "leakyrelu":
        return nn.LeakyReLU()
    if act == "sigmoid":
        return nn.Sigmoid()
    if act == "tanh":
        return nn.Tanh()
    if act == "selu":
        return nn.SELU()
    
    raise ValueError(f"Unknown activation function: {act}")



###########################################################################
###                          Model definition                           ###
###########################################################################



class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell.
    Parameters
    ----------
    input_dim: int
        Number of channels of input tensor.
    hidden_dim: int
        Number of channels of hidden state.
    kernel_size: int
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    ndim: int
        Number of dimensions
    batch_norm: bool
        Whether or not to apply batch normalization
    independent: bool
        Whether or not the output gate should be removed
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        kernel_size=3,
        bias=False,
        batch_norm=False,
        ndim=1,
        independent=False,
    ):

        super(ConvLSTMCell, self).__init__()

        if ndim not in [1, 2]:
            raise ValueError(f"Invalid number of dimensions: {ndim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.independent = independent
        self.batch_norm = batch_norm
        self.ndim = ndim

        # Initialize convolutional layers
        if self.independent:
            if ndim == 1:
                self.conv = nn.Conv1d(
                    in_channels=self.input_dim + self.hidden_dim,
                    out_channels=3 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    bias=self.bias,
                )
                self.pad = (self.padding, self.padding)
            elif ndim == 2:
                self.conv = nn.Conv2d(
                    in_channels=self.input_dim + self.hidden_dim,
                    out_channels=3 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    bias=self.bias,
                )
                self.pad = (self.padding, self.padding, self.padding, self.padding)
        else:
            if ndim == 1:
                self.conv = nn.Conv1d(
                    in_channels=self.input_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    bias=self.bias,
                )
                self.pad = (self.padding, self.padding)
            elif ndim == 2:
                self.conv = nn.Conv2d(
                    in_channels=self.input_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    bias=self.bias,
                )
                self.pad = (self.padding, self.padding, self.padding, self.padding)

        # Initialize batch normalization
        if self.batch_norm:
            if self.independent:
                if ndim == 1:
                    self.bn = nn.BatchNorm1d(3 * self.hidden_dim)
                elif ndim == 2:
                    self.bn = nn.BatchNorm2d(3 * self.hidden_dim)
            else:
                if ndim == 1:
                    self.bn = nn.BatchNorm1d(4 * self.hidden_dim)
                elif ndim == 2:
                    self.bn = nn.BatchNorm2d(4 * self.hidden_dim)

            if ndim == 1:
                self.bn_out = nn.BatchNorm1d(self.hidden_dim)
            elif ndim == 2:
                self.bn_out = nn.BatchNorm2d(self.hidden_dim)
        else:
            self.bn = nn.Identity()
            self.bn_out = nn.Identity()
        
        return



    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = nn.functional.pad(combined, self.pad, mode="replicate")
        combined_conv = self.conv(combined)

        combined_conv = self.bn(combined_conv)

        if self.independent:
            cc_i, cc_f, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            o = 1.

        else:
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            o = torch.sigmoid(cc_o)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(self.bn_out(c_next))

        return h_next, c_next



    def init_hidden(self, batch_size, size):

        if self.ndim == 1:
            h = torch.zeros(
                batch_size,
                self.hidden_dim,
                size,
                device=self.conv.weight.device
            )
        elif self.ndim == 2:
            h = torch.zeros(
                batch_size,
                self.hidden_dim,
                size[0],
                size[1],
                device=self.conv.weight.device
            )
        
        return h, h.clone()



class ConvLSTM(nn.Module):
    """
    Convolutional LSTM model
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        output_kernel_size: Size of output kernel
        output_act: Output activation function
        noise: Amount of noise to add to the input
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        kernel_size=3,
        num_layers=3,
        batch_input=1,
        bias=True,
        output_bias=True,
        batch_norm=False,
        ndim=1,
        independent=False,
        return_all_layers=False,
        output_kernel_size=1,
        output_act="sigmoid",
        noise=0.,
    ):
        super(ConvLSTM, self).__init__()

        if ndim not in [1, 2]:
            raise ValueError(f"Invalid number of dimensions: {ndim}")

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.is_ensemble = False
        self.noise = noise

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_input = batch_input
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_kernel_size = output_kernel_size
        self.output_padding = output_kernel_size // 2
        self.bias = bias
        self.output_bias = output_bias
        self.ndim = ndim
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = (
                self.input_dim * self.batch_input if i == 0 else self.hidden_dim[i - 1]
            )

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    batch_norm=batch_norm,
                    ndim=ndim,
                    independent=independent,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        if ndim == 1:
            self.output_conv = nn.Conv1d(
                in_channels=self.hidden_dim[-1],
                out_channels=1,
                kernel_size=self.output_kernel_size,
                #padding=self.output_padding,
                bias=self.output_bias,
            )
            self.out_pad = (self.output_padding, self.output_padding)
        elif ndim == 2:
            self.output_conv = nn.Conv2d(
                in_channels=self.hidden_dim[-1],
                out_channels=1,
                kernel_size=self.output_kernel_size,
                #padding=self.output_padding,
                bias=self.output_bias,
            )
            self.out_pad = (self.output_padding, self.output_padding, self.output_padding, self.output_padding)

        self.output_act = get_act(output_act)

        return



    def _batch_input(self, input_tensor):
        # Batch input layers
        cur_layer_input = []
        for i in range(input_tensor.size(1)):
            if i >= self.batch_input - 1:
                tmp_input = []
                for j in reversed(range(self.batch_input)):
                    tmp_input.append(torch.unsqueeze(input_tensor[:, i - j], 1))
                cur_layer_input.append(torch.cat(tmp_input, dim=2))
        return torch.cat(cur_layer_input, dim=1)



    def forward(self, input_tensor, hidden_state=None, repeat_first_input=1):
        """
        Parameters
        ----------
        input_tensor: todo
            4-D Tensor of shape (b, t, c, s)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """

        if self.ndim == 1:
            b, _, _, s = input_tensor.size()
            # Initialize hidden state
            if hidden_state is None:
                hidden_state = self._init_hidden(batch_size=b, image_size=s)
        
        elif self.ndim == 2:
            b, _, _, sx, sy = input_tensor.size()
            # Initialize hidden state
            if hidden_state is None:
                hidden_state = self._init_hidden(batch_size=b, image_size=[sx, sy])        

        layer_output_list = []
        last_state_list = []
        output_list = []

        # Batch input layers
        cur_layer_input = self._batch_input(input_tensor)

        if self.noise > 0.:
            cur_layer_input += torch.randn_like(cur_layer_input) * self.noise
        seq_len = cur_layer_input.size(1)

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            first_count = 1

            for t in range(seq_len):

                # Repeat the first input several times if required
                while first_count < repeat_first_input:
                    h, c = self.cell_list[layer_idx](
                        input_tensor=cur_layer_input[:, t], cur_state=[h, c]
                    )
                    output_inner.append(h)

                    if self.return_all_layers and layer_idx == self.num_layers - 1:
                        output = nn.functional.pad(h, self.out_pad, mode="replicate")
                        output = self.output_conv(output)
                        output = self.output_act(output)
                        output_list.append(output)
                
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t], cur_state=[h, c]
                )
                output_inner.append(h)

                if self.return_all_layers and layer_idx == self.num_layers - 1:
                    output = nn.functional.pad(h, self.out_pad, mode="replicate")
                    output = self.output_conv(output)
                    output = self.output_act(output)
                    output_list.append(output)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if self.return_all_layers:
            output = torch.cat(output_list, dim=1)
        else:
            output = nn.functional.pad(h, self.out_pad, mode="replicate")
            output = self.output_conv(output)
            output = self.output_act(output)

        return output, layer_output_list, last_state_list



    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states



    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class ConvLSTMEnsemble(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        n_models=4,
        hidden_dim=[64, 64, 64, 64],
        kernel_size=[3, 3, 3, 3],
        num_layers=3,
        batch_input=1,
        bias=True,
        output_bias=True,
        return_all_layers=False,
        batch_norm=False,
        ndim=1,
        independent=False,
        output_kernel_size=1,
        output_act="sigmoid",
        noise=0.,
        invert=False,
    ):
        super(ConvLSTMEnsemble, self).__init__()

        self.inv = invert
        self.is_ensemble = True
        self.noise = noise
        self.return_all_layers = return_all_layers
        self.ndim = ndim

        self.multiscale_ks = isinstance(kernel_size[0], list)
        self.multiscale_fks = isinstance(output_kernel_size, list)

        models = []
        for i in range(n_models):

            if self.multiscale_ks:
                ks = kernel_size[i]
            else:
                ks = kernel_size

            if self.multiscale_fks:
                fks = output_kernel_size[i]
            else:
                fks = output_kernel_size

            models.append(
                ConvLSTM(
                    input_dim,
                    hidden_dim,
                    ks,
                    num_layers,
                    batch_input=batch_input,
                    bias=bias,
                    output_bias=output_bias,
                    batch_norm=batch_norm,
                    ndim=ndim,
                    independent=independent,
                    return_all_layers=return_all_layers,
                    output_kernel_size=fks,
                    output_act=output_act,
                )
            )

        self.models = nn.ModuleList(models)

        return



    def forward(self, input_tensor, repeat_first_input=1):
        """
        Parameters
        ----------
        input_tensor: todo
            4-D Tensor of shape (b, t, c, s)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """

        if self.inv:
            input_tensor = input_tensor.flip(1)

        ys = []
        for net in self.models:
            if self.noise > 0.:
                X = input_tensor.clone()
                # Add noise only to the spectrum, not to the MAS encoding
                X[:, :, :-1] += torch.randn_like(input_tensor[:, :, :-1]) * self.noise
                y, _, _ = net(X, repeat_first_input=repeat_first_input)
            else:
                y, _, _ = net(input_tensor, repeat_first_input=repeat_first_input)
            ys.append(torch.unsqueeze(y.clone(), 0))

        ys = torch.cat(ys, dim=0)

        return torch.mean(ys, dim=0), torch.std(ys, dim=0), ys



###########################################################################
###                      Loss function definition                       ###
###########################################################################



class PIPLoss(nn.Module):
    """
    PIPNet custom loss
    """

    def __init__(
        self,
        trg_fuzz=0,
        trg_fuzz_len=25,
        ndim=1,
        exp=2.,
        offset=1.,
        factor=0.,
        int_w=0.,
        int_exp=2.,
        return_components=False,
        device="cpu",
    ):
        super(PIPLoss, self).__init__()

        self.trg_fuzz = trg_fuzz
        self.trg_fuzz_len = trg_fuzz_len
        self.ndim = ndim

        self.exp = exp
        self.offset = offset
        self.factor = factor

        self.int_w = int_w
        self.int_exp = int_exp

        self.return_components = return_components
        self.device = device

        if self.trg_fuzz > 0.:
            self.trg_filt = self.make_filt(self.trg_fuzz, self.trg_fuzz_len)

        return
    


    def make_filt(self, sig, l):

        pad = l // 2
        k = torch.exp(-torch.square(torch.arange(l) - pad) / (2 * (sig**2)))
        k /= torch.max(k)

        if self.ndim == 1:
            k = k.view(1, 1, l)

            filt = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=l,
                padding=pad,
                bias=False
            )

        elif self.ndim == 2:
            k = torch.outer(k, k)
            k = k.view(1, 1, l, l)

            filt = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=l,
                padding=pad,
                bias=False
            )
        
        filt.weight.data = k
        filt.weight.requires_grad = False
        filt.to(self.device)

        return filt
    


    def update_param(self, k, val):

        self.__setattr__(k, val)

        if self.trg_fuzz > 0.:
            self.trg_filt = self.make_filt(self.trg_fuzz, self.trg_fuzz_len)
        
        return
    


    def cmp_loss(self, y, y_trg):
        """
        'Comparison' loss: direct comparison between isotropic and predicted spectra
        """

        # Compute difference between output and target spectra
        x = torch.abs(y - y_trg)

        # Raise to the selected exponential
        x = torch.pow(x, self.exp)

        # Compute the scaling
        w = torch.ones_like(x) * self.offset

        w = torch.max(w, y_trg * self.factor)

        x = x * w

        return torch.mean(x)



    def int_loss(self, y, y_trg):
        """
        Intergral loss: Compare spectra integral
        """

        x = torch.mean(y, dim=-1) - torch.mean(y_trg, dim=-1)

        x = torch.pow(torch.abs(x), self.int_exp)

        return torch.mean(x) * self.int_w



    def __call__(self, y, y_trg):

        components = []
        tot_loss = 0.

        # Reshape to enable broadening filter application
        if self.ndim == 1:
            y = y.reshape(-1, 1, y.shape[-1])
            y_trg = y_trg.reshape(-1, 1, y_trg.shape[-1])

        elif self.ndim == 2:
            y = y.reshape(-1, 1, y.shape[-2], y.shape[-1])
            y_trg = y_trg.reshape(-1, 1, y_trg.shape[-2], y_trg.shape[-1])

        # Apply broadening filter
        if self.trg_fuzz > 0.:
            y_trg = self.trg_filt(y_trg)
        
        # Comparison loss
        tmp_loss = self.cmp_loss(y, y_trg)
        tot_loss += tmp_loss
        components.append(float(tmp_loss.detach().cpu()))

        # Integral loss
        if self.int_w > 0.:
            tmp_loss = self.int_loss(y, y_trg)
            tot_loss += tmp_loss
            components.append(float(tmp_loss.detach().cpu()))

        if self.return_components:
            return tot_loss, components

        return tot_loss
