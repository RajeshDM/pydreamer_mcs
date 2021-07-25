from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter


class GRU2Inputs(nn.Module):

    def __init__(self, input1_dim, input2_dim, mlp_dim=200, state_dim=200, num_layers=1, bidirectional=False, input_activation=F.elu):
        super().__init__()
        self._in_mlp1 = nn.Linear(input1_dim, mlp_dim)
        self._in_mlp2 = nn.Linear(input2_dim, mlp_dim, bias=False)
        self._act = input_activation
        self._gru = nn.GRU(input_size=mlp_dim, hidden_size=state_dim, num_layers=num_layers, bidirectional=bidirectional)
        self._directions = 2 if bidirectional else 1

    def init_state(self, batch_size):
        device = next(self._gru.parameters()).device
        return torch.zeros((
            self._gru.num_layers * self._directions,
            batch_size,
            self._gru.hidden_size), device=device)

    def forward(self,
                input1_seq: Tensor,  # (N,B,X1)
                input2_seq: Tensor,  # (N,B,X2)
                in_state: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        if in_state is None:
            in_state = self.init_state(input1_seq.size(1))
        inp = self._act(self._in_mlp1(input1_seq) + self._in_mlp2(input2_seq))
        output, out_state = self._gru(inp, in_state)
        # NOTE: Different from nn.GRU: detach output state
        return output, out_state.detach()


class GRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(input_size, 3 * hidden_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, 3 * hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates_i = torch.mm(input, self.weight_ih) + self.bias_ih
        gates_h = torch.mm(state, self.weight_hh) + self.bias_hh
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(reset_i + reset_h)
        update = torch.sigmoid(update_i + update_h)
        newval = torch.tanh(newval_i + reset * newval_h)

        h = update * newval + (1 - update) * state
        return h


class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size)
        self.ln_update = nn.LayerNorm(hidden_size)
        self.ln_newval = nn.LayerNorm(hidden_size)

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h


class NormGRUCellLateReset(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.lnorm = nn.LayerNorm(3 * hidden_size)
        self._update_bias = -1

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates = self.weight_ih(input) + self.weight_hh(state)
        gates = self.lnorm(gates)
        reset, update, newval = gates.chunk(3, 1)

        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update + self._update_bias)
        newval = torch.tanh(reset * newval)  # late reset, diff from normal GRU
        h = update * newval + (1 - update) * state
        return h

class LSTMCell(jit.ScriptModule):
    # Example from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)
