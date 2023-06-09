from torch import nn
import torch
from torch.autograd import Variable


# Bahdanau + location attention
class LocationAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        decoder_layers: int,
        nfilters: int,
        kernel_size: int,
        attention_smoothing: bool = False,
    ):
        super(LocationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_layers = decoder_layers

        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.conv1d = nn.Conv1d(1, nfilters, kernel_size, padding="same")
        self.prev_attn_proj = nn.Linear(nfilters, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

        if attention_smoothing:
            self.sigma = self._attn_smoothing
        else:
            self.sigma = nn.Softmax(dim=0)

    def _attn_smoothing(self, x):
        return self.sigmoid(x) / self.sigmoid(x).sum()

    # hidden:         layers, b, f
    # encoder_output: t, b, f
    # prev_attention: b, t
    def forward(self, hidden, encoder_output, enc_len, prev_attention):
        encoder_output = encoder_output.transpose(0, 1)  # b, t, f
        attn_energy = self.score(hidden, encoder_output, prev_attention)

        attn_weight = Variable(torch.zeros(attn_energy.shape)).cuda()
        for i, le in enumerate(enc_len):
            attn_weight[i, :le] = self.sigma(attn_energy[i, :le])
        return attn_weight.unsqueeze(2)

    # encoder_output: b, t, f
    def score(self, hidden, encoder_output, prev_attention):
        hidden = hidden.permute(1, 2, 0)  # b, f, layers
        addMask = torch.FloatTensor(
            [1 / self.decoder_layers] * self.decoder_layers
        ).view(1, self.decoder_layers, 1)
        addMask = torch.cat([addMask] * hidden.shape[0], dim=0)
        addMask = Variable(addMask.cuda())  # b, layers, 1
        hidden = torch.bmm(hidden, addMask)  # b, f, 1
        hidden = hidden.permute(0, 2, 1)  # b, 1, f
        hidden_attn = self.hidden_proj(hidden)  # b, 1, f

        prev_attention = prev_attention.unsqueeze(1)  # b, 1, t
        conv_prev_attn = self.conv1d(prev_attention)  # b, k, t
        conv_prev_attn = conv_prev_attn.permute(0, 2, 1)  # b, t, k
        conv_prev_attn = self.prev_attn_proj(conv_prev_attn)  # b, t, f

        encoder_output_attn = self.encoder_output_proj(encoder_output)
        res_attn = self.tanh(encoder_output_attn + hidden_attn + conv_prev_attn)
        out_attn = self.out(res_attn)  # b, t, 1
        out_attn = out_attn.squeeze(2)  # b, t
        return out_attn
