import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        super(MLP, self).__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, policy, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        policy = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return policy, logp_a


class CategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(CategoricalActor, self).__init__()
        self.logits_net = MLP([obs_dim] + list(hidden_sizes) + [act_dim])

    def forward(self, obs, act=None):
        policy = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(policy, act)
        return policy, logp_a

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, policy, act):
        return policy.log_prob(act)


class NormalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(NormalActor, self).__init__()
        self.mu_net = MLP([obs_dim] + list(hidden_sizes) + [act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, dtype=torch.float32))

    def forward(self, obs, act=None):
        policy = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(policy, act)
        return policy, logp_a

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        return Normal(mu, self.log_std.exp())

    def _log_prob_from_distribution(self, policy, act):
        return policy.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        self.v_net = MLP([obs_dim] + list(hidden_sizes) + [1])

    def forward(self, obs):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
