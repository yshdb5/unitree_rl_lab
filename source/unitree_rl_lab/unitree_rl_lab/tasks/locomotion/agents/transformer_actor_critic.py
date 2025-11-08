import math, torch
import torch.nn as nn

def _diag_gauss(mean, log_std):
    std = torch.exp(log_std).expand_as(mean)
    return torch.distributions.Normal(mean, std)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):  # [B,L,D]
        return x + self.pe[:, :x.size(1), :]

class TransformerActorCritic(nn.Module):
    """
    Student π_o: causal transformer over L=context_len steps.
    Expects obs already stacked: [B, L*D_step]. We infer D_step at build.
    """
    def __init__(self, obs_dim: int, action_dim: int,
                 context_len=16, d_model=192, nhead=4, nlayers=4, token_mlp=512):
        super().__init__()
        assert obs_dim % context_len == 0, "obs_dim must be divisible by context_len"
        self.L = context_len
        self.Ds = obs_dim // context_len

        self.token_proj = nn.Sequential(
            nn.Linear(self.Ds, token_mlp), nn.ReLU(),
            nn.Linear(token_mlp, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pos = PositionalEncoding(d_model, max_len=context_len)

        self.pi = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(),
                                nn.Linear(256, action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.vf = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(),
                                nn.Linear(256, 1))

    def act(self, obs):
        dist, _ = self._dist(obs)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1)

    def evaluate_actions(self, obs, actions):
        dist, h = self._dist(obs)
        logp = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        v = self.vf(h).squeeze(-1)
        return logp, ent, v, dist

    def value(self, obs):
        _, h = self._dist(obs)
        return self.vf(h).squeeze(-1)

    def _dist(self, obs_flat):
        B = obs_flat.size(0)
        x = obs_flat.view(B, self.L, self.Ds)
        x = self.token_proj(x)
        x = self.pos(x)
        L = x.size(1)
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.encoder(x, mask=causal)[:, -1, :]   # last token
        mean = self.pi(h)
        return _diag_gauss(mean, self.log_std), h
