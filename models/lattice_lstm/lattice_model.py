import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np


class WordLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        self.weight_hh.data.copy_(torch.eye(self.hidden_size).repeat(1, 3))
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, input_, hx):
        h_0, c_0 = hx
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)

        batch_size = h_0.size(0)
        wh_b = torch.addmm(self.bias.unsqueeze(0).expand(batch_size, -1), h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        return c_1


class MultiInputLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))

        self.alpha_weight_ih = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.alpha_weight_ih.data)
        self.weight_hh.data.copy_(torch.eye(self.hidden_size).repeat(1, 3))
        self.alpha_weight_hh.data.copy_(torch.eye(self.hidden_size))
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)
            nn.init.constant_(self.alpha_bias.data, 0)

    def forward(self, input_, c_input, hx):
        h_0, c_0 = hx
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)

        batch_size = h_0.size(0)
        wh_b = torch.addmm(self.bias.unsqueeze(0).expand(batch_size, -1), h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)

        i, o, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        if not c_input:
            c_1 = (1 - i) * c_0 + i * g
        else:
            c_input_var = torch.cat(c_input, dim=0).squeeze(1)
            alpha_wi = torch.mm(input_, self.alpha_weight_ih).expand(len(c_input), -1)
            alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)
            alpha = torch.sigmoid(alpha_wi + alpha_wh)

            alpha = torch.exp(torch.cat([i, alpha], dim=0))
            alpha = alpha / (alpha.sum(dim=0, keepdim=True) + 1e-12)

            merge = torch.cat([g, c_input_var], dim=0)
            c_1 = (merge * alpha).sum(dim=0, keepdim=True)

        h_1 = o * torch.tanh(c_1)
        return h_1, c_1


class LatticeLSTM(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, hidden_dim, word_drop,
                 word_alphabet_size, word_emb_dim, num_labels,
                 pretrain_word_emb=None, fix_word_emb=True, gpu=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gpu = gpu

        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -np.sqrt(3.0 / char_emb_dim), np.sqrt(3.0 / char_emb_dim))

        self.word_emb = nn.Embedding(word_alphabet_size, word_emb_dim, padding_idx=0)
        if pretrain_word_emb is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(pretrain_word_emb).float())
        else:
            nn.init.uniform_(self.word_emb.weight, -np.sqrt(3.0 / word_emb_dim), np.sqrt(3.0 / word_emb_dim))
        if fix_word_emb:
            self.word_emb.weight.requires_grad = False

        self.word_dropout = nn.Dropout(p=word_drop)
        self.rnn = MultiInputLSTMCell(input_size=char_emb_dim, hidden_size=hidden_dim)
        self.word_rnn = WordLSTMCell(input_size=word_emb_dim, hidden_size=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        if self.gpu and torch.cuda.is_available():
            self.cuda()

    def forward(self, input_seq, skip_input_list, hidden=None):
        skip_input = skip_input_list[0]
        char_emb = self.char_emb(input_seq).transpose(1, 0)  # (seq_len, 1, char_emb_dim)

        seq_len = char_emb.size(0)
        batch_size = char_emb.size(1)
        assert batch_size == 1

        if hidden is None:
            hx = torch.zeros(batch_size, self.hidden_dim).to(char_emb.device)
            cx = torch.zeros(batch_size, self.hidden_dim).to(char_emb.device)
        else:
            hx, cx = hidden

        hidden_out = []
        memory_out = []
        input_c_list = [[] for _ in range(seq_len)]

        for t in range(seq_len):
            current_char = char_emb[t].squeeze(0)
            hx, cx = self.rnn(current_char, input_c_list[t], (hx, cx))
            hidden_out.append(hx)
            memory_out.append(cx)

            if skip_input[t]:
                word_ids, word_lengths = skip_input[t]
                word_var = torch.LongTensor(word_ids).to(char_emb.device)
                word_emb = self.word_emb(word_var)
                word_emb = self.word_dropout(word_emb)

                ct = self.word_rnn(word_emb, (hx, cx))
                for idx, length in enumerate(word_lengths):
                    end_pos = t + length - 1
                    if 0 <= end_pos < seq_len:
                        input_c_list[end_pos].append(ct[idx:idx + 1])

        output_hidden = torch.stack(hidden_out, dim=0).unsqueeze(0)
        logits = self.classifier(output_hidden)
        return logits