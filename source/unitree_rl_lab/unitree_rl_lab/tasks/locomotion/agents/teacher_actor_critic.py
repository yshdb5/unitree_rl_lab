import torch, torch.nn as nn

def _diag_gauss(mean, log_std):
    std = torch.exp(log_std).expand_as(mean)
    return torch.distributions.Normal(mean, std)

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden=(512,512,256,128)):
        super().__init__()
        f = []
        last = obs_dim
        for h in hidden:
            f += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.pi = nn.Sequential(*f, nn.Linear(last, action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.vf = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256,1))

    def act(self, obs):
        dist = _diag_gauss(self.pi(obs), self.log_std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1)

    def evaluate_actions(self, obs, actions):
        dist = _diag_gauss(self.pi(obs), self.log_std)
        logp = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        v = self.vf(obs).squeeze(-1)
        return logp, ent, v, dist

    def value(self, obs): return self.vf(obs).squeeze(-1)
