import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, GRU, Sigmoid, LSTM
import torch.distributions as tdt
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            # nn.Softmax(dim=1),
        )
        # self.sf_layer = F.softmax(dim=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        # x = self.sf_layer(x)
        return F.softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=12, out_features=64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(0)
        # x should be of shape (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # We use the output from the last timestep
        out = self.fc(out[:, -1, :])
        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):

        seq_len = encoder_outputs.size(1)

        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(encoder_outputs + hidden_expanded))  # [batch, seq_len, hidden_dim]
        energy = energy @ self.v  # [batch, seq_len]

        attn_weights = F.softmax(energy, dim=1)  # [batch, seq_len]

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, hidden_dim]

        return context, attn_weights


class AttGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=0.0):
        super(AttGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)

        self.attn = nn.Linear(hidden_size * self.num_directions, 1, bias=False)

        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        # x: [seq_len, batch, input_size]
        gru_out, _ = self.gru(x)  # [seq_len, batch, hidden_size * num_directions]

        attn_weights = torch.tanh(self.attn(gru_out))  # [seq_len, batch, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        context = gru_out * attn_weights  # [seq_len, batch, hidden_size * num_directions]

        output = self.fc(context)  # [seq_len, batch, output_size]
        return output, attn_weights


class Proposal_network(torch.nn.Module):
    def __init__(self, args):
        super(Proposal_network, self).__init__()
        self.num_features = args.num_features
        self.context_units = args.context_units
        self.mixture_components = args.mixture_components
        self.residual_blocks = args.residual_blocks
        self.hidden_units = args.hidden_units

        self.fc1 = torch.nn.Sequential(
                        Linear(self.num_features*2, self.hidden_units))

        self.fc2 = torch.nn.Sequential(
                        Linear(self.hidden_units, self.hidden_units),
                        ReLU(),
                        # Dropout(p=0.1),
                        Linear(self.hidden_units, self.hidden_units))

        self.fc3 = torch.nn.Sequential(
                        ReLU(),
                        Linear(self.hidden_units, self.num_features * (3 * self.mixture_components + self.context_units)))

    def create_proposal_dist(self, t):
        logits = t[..., :self.mixture_components]
        means = t[..., self.mixture_components:-self.mixture_components]
        scales = F.softplus(t[..., -self.mixture_components:]) + 1e-3
        components_dist = tdt.Normal(
            loc=means.to(torch.float32), scale=scales.to(torch.float32)
        )
        return tdt.MixtureSameFamily(
            mixture_distribution=tdt.Categorical(
                logits=logits.to(torch.float32)
            ),
            component_distribution=components_dist,
        )

    def forward(self, x, observed_mask):

        x = x * observed_mask
        x_o = x

        h = torch.concat([x_o, observed_mask], dim=1)
        h = self.fc1(h)
        for _ in range(self.residual_blocks):
            res = self.fc2(h)
            h = torch.add(h, res)
        h = self.fc3(h)
        h = torch.reshape(h, [-1, self.num_features, 3 * self.mixture_components + self.context_units])

        context = h[..., :self.context_units]
        params = h[..., self.context_units:]

        proposal_dist = self.create_proposal_dist(params)

        return proposal_dist, context


class Energy_network(torch.nn.Module):
    def __init__(self, args,):
        super(Energy_network, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.context_units = args.context_units
        self.residual_blocks = args.residual_blocks
        self.hidden_units = args.hidden_units
        self.energy_clip = args.energy_clip

        self.fc1 = torch.nn.Sequential(
                        Linear(1+self.num_features+self.context_units, self.hidden_units))

        self.fc2 = torch.nn.Sequential(
                        Linear(self.hidden_units, self.hidden_units),
                        ReLU(),
                        Dropout(args.dropout),
                        Linear(self.hidden_units, self.hidden_units))

        self.fc3 = torch.nn.Sequential(
                        ReLU(),
                        Dropout(args.dropout),
                        Linear(self.hidden_units, 1),
                        Sigmoid())

    def forward(self, x_u_i, u_i, context):
        u_i_one_hot = torch.nn.functional.one_hot(u_i.to(torch.int64), self.num_features).to(self.args.device)
        h = torch.concat([torch.unsqueeze(x_u_i.clone(), dim=-1), u_i_one_hot.clone(), context.clone()], dim=-1)

        h = self.fc1(h)
        for _ in range(self.residual_blocks):
            res = self.fc2(h)
            h = torch.add(h, res)
        h = self.fc3(h)

        energies = F.softplus(h)
        energies = torch.clip_(energies, 0.0, self.energy_clip)
        negative_energies = -energies

        return negative_energies

class Gru_proposal_network(torch.nn.Module):
    def __init__(self, args):
        super(Gru_proposal_network, self).__init__()
        self.num_features = args.num_features
        self.num_seq = args.num_seq
        self.gru_units = args.proposal_unit
        self.context_units = args.context_units
        self.mixture_components = args.mixture_components
        self.hidden_units = args.hidden_units
        self.args = args

        if self.args.proposal_network == 'GRU':
            self.gru_layer = GRU(self.num_features*2,
                                 self.num_features *
                                 (2 * self.mixture_components + self.context_units))

        if self.args.proposal_network == 'BiGRU':
            self.gru_layer = GRU(self.num_features*2,
                                 self.num_features *
                                 (2 * self.mixture_components + self.context_units),
                                 bidirectional=True)

        if self.args.proposal_network == 'LSTM':
            self.gru_layer = LSTM(self.num_features*2,
                                 self.num_features *
                                 (2 * self.mixture_components + self.context_units))

        if self.args.proposal_network == 'AttGRU':
            self.gru_layer = AttGRU(self.num_features*2, 16,
                                 self.num_features *
                                 (2 * self.mixture_components + self.context_units))
    def create_proposal_dist(self, t):
        # logits = t[..., :self.mixture_components]
        # means = t[..., self.mixture_components:-self.mixture_components]
        # scales = F.softplus(t[..., -self.mixture_components:]) + 1e-3
        means = t[..., :self.mixture_components].squeeze(-1)
        scales = F.softplus(t[..., -self.mixture_components:].squeeze(-1)) + 1e-3
        components_dist = tdt.Normal(
            loc=means.to(torch.float32), scale=scales.to(torch.float32)
        )
        # return tdt.MixtureSameFamily(
        #     mixture_distribution=tdt.Categorical(
        #         logits=logits.to(torch.float32)
        #     ),
        #     component_distribution=components_dist,
        # )

        return components_dist

    def forward(self, data, observed_mask):
        # h = torch.concatenate([x_o, observed_mask], dim=1)
        # hat = -1 * torch.ones(batch_size, 1).cuda()
        if self.args.if_seq == False:
            phi = torch.concat([data, observed_mask], dim=-1)
            # phi = torch.concat([hat, phi], dim=1)
            # phi = torch.unsqueeze(phi.to(torch.float32), 1)
            h, c = self.gru_layer(phi)
        else:
            batch_size, seq_len, num_features = data.shape
            x_o = data * observed_mask
            c = None
            for t in range(seq_len):
                for f in range(num_features):
                    if observed_mask[:, t, f].sum() < batch_size:
                        mask_now = observed_mask[:, t, f] == 1
                        x_o[mask_now, t, f] = data[mask_now, t, f]
                    phi_t = torch.cat([x_o[:, t, :], observed_mask[:, t, :]], dim=-1).unsqueeze(1)
                    h, c = self.gru_layer(phi_t, c)

        h = F.relu(h[:, :, -(self.num_features*(2*self.mixture_components+self.context_units)):])

        h = torch.reshape(h, [-1, self.num_seq, self.num_features, 2 * self.mixture_components + self.context_units])

        context = h[..., :self.context_units]
        params = h[..., self.context_units:]
        proposal_dist = self.create_proposal_dist(params)

        return proposal_dist, context