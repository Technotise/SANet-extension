import torch
import random
from torch import nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim, decoder_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, hidden, encoder_output):
        src_len = encoder_output.shape[1]
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_output), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class SeqDecode(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(SeqDecode, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=input_size - 3)
        self.rnn = nn.LSTM(hidden_size * 3, hidden_size, n_layers, batch_first=True)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size * 4, input_size)
        self.dropout = nn.Dropout(0, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0.data)
                nn.init.orthogonal_(m.weight_hh_l0.data)
                m.bias_ih_l0.data.fill_(0)
                m.bias_hh_l0.data.fill_(0)

    def forward(self, x, h, c, encoder_output):
        self.rnn.flatten_parameters()
        x = x.unsqueeze(1)
        embed = self.dropout(self.embedding(x))
        att = self.attention(h, encoder_output).unsqueeze(1)
        att = torch.bmm(att, encoder_output)
        if embed.size(0) != att.size(0):
            min_batch_size = min(embed.size(0), att.size(0))
            embed = embed[:min_batch_size]
            att = att[:min_batch_size]
            h = h[:, :min_batch_size, :]
            c = c[:, :min_batch_size, :]
        if embed.size(1) != att.size(1):
            min_seq_len = min(embed.size(1), att.size(1))
            embed = embed[:, :min_seq_len, :]
            att = att[:, :min_seq_len, :]
        x = torch.cat((embed, att), dim=2)
        x, (h, c) = self.rnn(x, (h, c))
        x = self.out(torch.cat((x, att, embed), dim=2)).squeeze(1)
        return x, h, c

class LSTM_seq(nn.Module):
    def __init__(self, max_seq=10, input_size=4096, hidden_size=4096, class_num=10, device=1):
        super(LSTM_seq, self).__init__()
        self.max_seq = max_seq
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.device = device
        self.Bilstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.Bilstm2 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.SeqDecode = SeqDecode(self.class_num, hidden_size, n_layers=1).to(self.device)
        self.memory_cell1 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.memory_cell2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.memory_cell3 = nn.Linear(hidden_size * 2, hidden_size).to(self.device)
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0.data)
                nn.init.orthogonal_(m.weight_ih_l0.data)
                m.bias_ih_l0.data.fill_(0)
                m.bias_hh_l0.data.fill_(0)

    def forward(self, x, label):
        x = x.to(self.device)
        label = label.to(self.device)
        self.Bilstm1.flatten_parameters()
        self.Bilstm2.flatten_parameters()
        self.lstm.flatten_parameters()
        batch, _, _ = x.shape
        x, (h, c) = self.Bilstm1(x)
        h = torch.tanh(self.memory_cell1(h))
        c = torch.tanh(self.memory_cell1(c))
        x, (h, c) = self.Bilstm2(x, (h, c))
        h = torch.tanh(self.memory_cell2(h))
        c = torch.tanh(self.memory_cell2(c))
        x, (h, c) = self.lstm(x, (h, c))
        h = torch.tanh(self.memory_cell3(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))).unsqueeze(0)
        c = torch.tanh(self.memory_cell3(torch.cat((c[-2, :, :], c[-1, :, :]), dim=1))).unsqueeze(0)
        w = torch.full(size=(batch,), fill_value=(self.class_num - 2), dtype=torch.long).to(self.device)
        outputs = torch.zeros(self.max_seq, batch, self.class_num).to(self.device)
        for t in range(self.max_seq):
            if w.size(0) != x.size(0):
                min_batch_size = min(w.size(0), x.size(0))
                w = w[:min_batch_size]
                h = h[:, :min_batch_size, :]
                c = c[:, :min_batch_size, :]
                x = x[:min_batch_size]
            w, h, c = self.SeqDecode(w, h, c, x)
            outputs[t, :w.size(0), :] = w
            teacher_force = random.random() < 0.5
            top1 = w.max(1)[1].detach()
            if teacher_force and t < label.size(0):
                w = label[t]
            else:
                w = top1
        return outputs.permute(1, 0, 2)