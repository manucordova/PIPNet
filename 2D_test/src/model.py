###########################################################################
###                               PIPNet                                ###
###                          Model definition                           ###
###                        Author: Manuel Cordova                       ###
###       Adapted from https://github.com/ndrplz/ConvLSTM_pytorch       ###
###                       Last edited: 2021-09-24                       ###
###########################################################################

import numpy as np
import torch
import torch.nn as nn



class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        bias,
        independent=False,
    ):
        """
        Initialize ConvLSTM cell.
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
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.independent = independent

        if self.independent:
            self.conv = nn.Conv2d(
                in_channels=self.input_dim + self.hidden_dim,
                out_channels=3 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=self.input_dim + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            )

    def analyze(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        combined_conv = self.bn(combined_conv)

        if self.independent:
            cc_i, cc_f, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            o = 1.0

        else:
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            o = torch.sigmoid(cc_o)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return i, f, o, g, h_next, c_next

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        if self.independent:
            cc_i, cc_f, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            o = 1.0

        else:
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            o = torch.sigmoid(cc_o)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, size_x, size_y):
        return (
            torch.zeros(
                batch_size, self.hidden_dim, size_x, size_y, device=self.conv.weight.device
            ),
            torch.zeros(
                batch_size, self.hidden_dim, size_x, size_y, device=self.conv.weight.device
            ),
        )


class ConvLSTM(nn.Module):

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
        hidden_dim,
        kernel_size,
        num_layers,
        batch_input=1,
        bias=True,
        final_bias=True,
        independent=False,
        return_all_layers=False,
        final_kernel_size=1,
        final_act="sigmoid",
        noise=0.0,
    ):
        super(ConvLSTM, self).__init__()

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
        self.final_kernel_size = final_kernel_size
        self.final_padding = final_kernel_size // 2
        self.bias = bias
        self.final_bias = final_bias
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
                    independent=independent,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        self.final_conv = nn.Conv2d(
            in_channels=self.hidden_dim[-1],
            out_channels=1,
            kernel_size=self.final_kernel_size,
            padding=self.final_padding,
            bias=self.final_bias,
        )

        if final_act == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_act == "softplus":
            self.final_act = nn.Softplus()
        elif final_act == "linear":
            self.final_act = nn.Identity()
        else:
            raise ValueError(f"Unknown final activation: {final_act}")

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

    def forward(self, input_tensor, hidden_state=None):
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

        b, _, _, sx, sy = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size_x=sx, image_size_y=sy)

        layer_output_list = []
        last_state_list = []
        output_list = []

        # Batch input layers
        cur_layer_input = self._batch_input(input_tensor)

        if self.noise > 0.0:
            cur_layer_input += torch.randn_like(cur_layer_input) * self.noise
        seq_len = cur_layer_input.size(1)

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

                if self.return_all_layers and layer_idx == self.num_layers - 1:
                    output = self.final_conv(h)
                    output = self.final_act(output)
                    output_list.append(output)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if self.return_all_layers:
            output = torch.cat(output_list, dim=1)
        else:
            output = self.final_conv(h)
            output = self.final_act(output)

        return output, layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size_x, image_size_y):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size_x, image_size_y))
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
        n_models,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_input=1,
        bias=True,
        final_bias=True,
        return_all_layers=False,
        independent=False,
        final_kernel_size=1,
        final_act="sigmoid",
        noise=0.0,
    ):
        super(ConvLSTMEnsemble, self).__init__()

        self.is_ensemble = True
        self.noise = noise
        self.return_all_layers = return_all_layers

        self.multiscale_ks = isinstance(kernel_size[0], list)
        self.multiscale_fks = isinstance(final_kernel_size, list)

        models = []
        for i in range(n_models):

            if self.multiscale_ks:
                ks = kernel_size[i]
            else:
                ks = kernel_size

            if self.multiscale_fks:
                fks = final_kernel_size[i]
            else:
                fks = final_kernel_size

            models.append(
                ConvLSTM(
                    input_dim,
                    hidden_dim,
                    ks,
                    num_layers,
                    batch_input=batch_input,
                    bias=bias,
                    final_bias=final_bias,
                    independent=independent,
                    return_all_layers=return_all_layers,
                    final_kernel_size=fks,
                    final_act=final_act,
                )
            )

        self.models = nn.ModuleList(models)

        return

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            4-D Tensor of shape (b, t, c, sx, sy)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """

        ys = []
        for net in self.models:
            if self.noise > 0.0:
                X = input_tensor.clone()
                # Add noise only to the spectrum, not to the MAS encoding
                X[:, :, :-1] + torch.randn_like(input_tensor[:, :, :-1]) * self.noise
                y, _, _ = net(X)
            else:
                y, _, _ = net(input_tensor)
            ys.append(torch.unsqueeze(y.clone(), 0))

        ys = torch.cat(ys, dim=0)

        return torch.mean(ys, dim=0), torch.std(ys, dim=0), ys


class CustomLoss(nn.Module):
    def __init__(
        self,
        trg_fuzz=0,
        trg_fuzz_len=25,
        srp_w=1.0,
        srp_exp=2.0,
        srp_offset=1.0,
        srp_fac=0.0,
        brd_w=0.0,
        brd_sig=5,
        brd_len=25,
        brd_exp=2.0,
        brd_offset=1.0,
        brd_fac=0.0,
        int_w=0.0,
        int_exp=2.0,
        int_edge=0,
        return_components=False,
        device="cpu",
    ):
        super(CustomLoss, self).__init__()

        self.trg_fuzz = trg_fuzz
        self.trg_fuzz_len = trg_fuzz_len

        self.srp_w = srp_w
        self.srp_exp = srp_exp
        self.srp_offset = srp_offset
        self.srp_fac = srp_fac

        self.brd_w = brd_w
        self.brd_exp = brd_exp
        self.brd_offset = brd_offset
        self.brd_fac = brd_fac
        self.brd_sig = brd_sig
        self.brd_len = brd_len

        self.int_w = int_w
        self.int_exp = int_exp
        self.int_edge = int_edge

        self.return_components = return_components
        self.device = device

        if srp_w == 0.0 and brd_w == 0.0:
            raise ValueError("At least one of the loss weights should be non-zero!")
        
        if self.trg_fuzz > 0.:

            self.trg_filt = self.make_filt(self.trg_fuzz, self.trg_fuzz_len)

        if self.brd_w > 0.0:

            self.brd_filt = self.make_filt(self.brd_sig, self.brd_len)

        return
    
    def make_filt(self, sig, l):

        pad = l // 2
        k = (
            1.0
            / (2 * np.pi * (sig ** 2))
            * torch.exp(
                -torch.square(torch.arange(l) - pad)
                / (2 * (sig ** 2))
            )
        )
        k = torch.outer(k, k)
        k /= torch.max(k)
        k = k.view(1, 1, l, l)


        filt = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=l,
            padding=pad,
            bias=False,
        )
        filt.weight.data = k
        filt.weight.requires_grad = False
        filt.to(self.device)

        return filt
    
    def update_param(self, k, val):

        self.__setattr__(k, val)

        if self.trg_fuzz > 0.:

            self.trg_filt = self.make_filt(self.trg_fuzz, self.trg_fuzz_len)

        if self.brd_w > 0.0:

            self.brd_filt = self.make_filt(self.brd_sig, self.brd_len)
        
        return

    def srp_loss(self, y, y_trg):
        """
        'Sharp' loss: direct comparison between isotropic and predicted spectra
        """

        # Compute difference between output and target spectra
        x = torch.abs(y - y_trg)

        # Raise to the selected exponential
        x = torch.pow(x, self.srp_exp)

        # Compute the scaling
        w = torch.ones_like(x) * self.srp_offset

        if self.srp_fac > 0.0:
            w = torch.max(w, y_trg * self.srp_fac)

        x = x * w

        return torch.mean(x) * self.srp_w

    def brd_loss(self, y, y_trg, return_specs=False):
        """
        'Broad' loss: comparison between broadened isotropic and predicted spectra
        """

        # Perform 2D convolution
        y2 = self.brd_filt(y)

        # Perform 2D convolution
        y2_trg = self.brd_filt(y_trg)

        if return_specs:
            return y2, y2_trg

        # Compute difference between output and target spectra
        x = torch.abs(y2 - y2_trg)

        # Raise to the selected exponential
        x = torch.pow(x, self.brd_exp)

        # Compute the scaling
        w = torch.ones_like(x) * self.brd_offset

        if self.brd_fac > 0.0:
            w = torch.max(w, y2_trg * self.brd_fac)

        x = x * w

        return torch.mean(x) * self.brd_w

    def int_loss(self, y, y_trg):
        """
        Intergral loss: Compare spectra integral
        """

        if self.int_edge > 0:

            x = torch.mean(y[..., self.int_edge:-self.int_edge], dim=-1) - torch.mean(y_trg[..., self.int_edge:-self.int_edge], dim=-1)
            x = torch.mean(x[..., self.int_edge:-self.int_edge], dim=-1)
        
        else:

            x = torch.mean(y, dim=-1) - torch.mean(y_trg, dim=-1)
            x = torch.mean(x, dim=-1)

        x = torch.pow(torch.abs(x), self.int_exp)

        return torch.mean(x) * self.int_w

    def __call__(self, y, y_trg):

        components = []
        tot_loss = 0.0

        y = y.reshape(-1, 1, y.shape[-2], y.shape[-1])
        y_trg = y_trg.reshape(-1, 1, y_trg.shape[-2], y_trg.shape[-1])        

        if self.trg_fuzz > 0.:
            y_trg = self.trg_filt(y_trg)

        if self.srp_w > 0.0:
            tmp_loss = self.srp_loss(y, y_trg)
            tot_loss += tmp_loss
            components.append(float(tmp_loss.detach().cpu()))

        if self.brd_w > 0.0:
            tmp_loss = self.brd_loss(y, y_trg)
            tot_loss += tmp_loss
            components.append(float(tmp_loss.detach().cpu()))

        if self.int_w > 0.0:
            tmp_loss = self.int_loss(y, y_trg)
            tot_loss += tmp_loss
            components.append(float(tmp_loss.detach().cpu()))

        if self.return_components:

            return tot_loss, components

        return tot_loss
