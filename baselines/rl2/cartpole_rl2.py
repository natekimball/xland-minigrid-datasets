import os
import gym
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math

from tqdm import tqdm

from src.model_tuples_cache import Transformer, KVCache
from src.nn import TransformerBlock, ObservationEncoder

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class CartpoleRL2(nn.Module):
    def __init__(
            self,
            num_actions: int,
            observation_dim: int,
            seq_len: int = 200,
            embedding_dim: int = 64,
            hidden_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 4,
            attention_dropout: float = 0.5,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.1,
            normalize_qk: bool = False,
            pre_norm: bool = True
            ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.seq_len = seq_len

        self.emb_dropout = nn.Dropout(embedding_dropout)

        self.obs_encoder = nn.Linear(
            observation_dim, embedding_dim
        )
        
        self.embed_transition = nn.Linear(
            embedding_dim + num_actions + 1, # [state, action, reward]
            hidden_dim
        )

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    normalize_qk=normalize_qk,
                    pre_norm=pre_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

        self.to(self.device)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        if isinstance(module, nn.Conv2d):
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(module.weight.data, gain)
            if hasattr(module.bias, "data"):
                module.bias.data.fill_(0.0)
    
    def forward(self,
                query_observations: torch.Tensor, # [batch_size, 4] or [batch_size, 4]
                context_observations: torch.Tensor, # [batch_size, seq_len, 4] or [batch_size, seq_len, 4]
                context_actions: torch.Tensor, # [batch_size, seq_len]
                # context_next_observations: torch.Tensor, # [batch_size, seq_len, 4] or [batch_size, seq_len, 4]
                context_rewards: torch.Tensor, # [batch_size, seq_len]
                ) -> torch.Tensor:
        # TODO: If I can't do positional encoding, I need to encode entire transition with next state
        # TODO: investigate AD to see how they handle positional encoding
        if context_observations.numel() == 0:
            batch_size, seq_len = query_observations.shape[0], 0
            # [batch_size, 1, embedding_dim]
            query_obs_emb = self.obs_encoder(query_observations.unsqueeze(1))

            zeros = torch.zeros(batch_size, 1, self.num_actions + 1, device=query_observations.device, dtype=query_observations.dtype)

            sequence = torch.cat(
                [query_obs_emb, zeros], dim=-1
            )  # [batch_size, seq_len + 1, state_embedding_dim + num_actions + 1]

        else:
            batch_size, seq_len = context_rewards.shape[0], context_rewards.shape[1]
            
            # [batch_size, 1, embedding_dim]
            query_obs_emb = self.obs_encoder(query_observations.unsqueeze(1))
            
            # [batch_size, seq_len, embedding_dim]
            context_obs_emb = self.obs_encoder(context_observations)

            # [batch_size, seq_len, num_actions]
            context_actions_emb = F.one_hot(context_actions, num_classes=self.num_actions)

            zeros = torch.zeros(batch_size, 1, device=context_observations.device, dtype=context_observations.dtype)
            
            # print(context_observations.shape, query_observations.shape)
            # print('emb', context_obs_emb.shape, query_obs_emb.shape)

            # [batch_size, seq_len + 1, embedding_dim]
            observation_seq = torch.cat([context_obs_emb, query_obs_emb], dim=1)

            action_seq = torch.cat(
                [context_actions_emb,
                zeros.to(context_actions.dtype).unsqueeze(-1).repeat_interleave(self.num_actions, dim=-1)],
                dim=1
            )

            # [batch_size, seq_len + 1]
            reward_seq = torch.cat(
                [context_rewards,
                zeros.to(context_rewards.dtype)],
                dim=1
            ).unsqueeze(-1)

            # print(observation_seq.shape, action_seq.shape, reward_seq.shape)

            sequence = torch.cat(
                [observation_seq, action_seq, reward_seq], dim=-1
            )  # [batch_size, seq_len + 1, state_embedding_dim + num_actions + 1]


        sequence = self.embed_transition(sequence)
        sequence = self.pos_encoder(sequence)

        out = self.emb_dropout(sequence)
        for block in self.blocks:
            out = block(out)

        # print(sequence.shape)
        # out = self.transformer(sequence, sequence)
        # print(out.shape)
        
        last_out = out[:, -1, :]
        # print(f'{last_out=}')

        # [batch_size, seq_len + 1, num_actions]
        logits = self.action_head(last_out)
        policy = F.softmax(logits, dim=-1)

        # [batch_size, seq_len + 1, 1]
        values = self.value_head(last_out)

        return policy, values

gamma = 0.99
lambda_gae = 0.95
eps_clip = 0.2
learning_rate = 3e-4
episodes = 10000
ppo_epochs = 5
batch_size = 64
seq_len = 4096

env = gym.make("CartPole-v1")

model = CartpoleRL2(
    num_actions=env.action_space.n,
    observation_dim=env.observation_space.shape[0],
    embedding_dim=128,
    hidden_dim=256,
    seq_len=seq_len,
    num_layers=2,
    num_heads=2,
).to(torch.bfloat16)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def compute_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = [None]*len(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.bfloat16), torch.tensor(returns, dtype=torch.bfloat16)

def rollout(model, env):
    model.eval()
    state, info = env.reset()
    # cache = model.init_cache(batch_size=1, dtype=torch.bfloat16, device="cpu")
    log_probs = []
    values = []
    rewards = []
    dones = []
    states = []
    actions = []

    with torch.no_grad():
        for t in range(seq_len):
            state = torch.tensor(state, device=model.device, dtype=torch.bfloat16)
            if states:
                policy, value = model(
                    state.unsqueeze(0),
                    torch.stack(states).unsqueeze(0),
                    torch.tensor(actions, device=model.device, dtype=torch.long).unsqueeze(0),
                    torch.tensor(rewards, device=model.device, dtype=torch.bfloat16).unsqueeze(0)
                )
            else:
                policy, value = model(
                    state.unsqueeze(0),
                    torch.empty(1, 0, *env.observation_space.shape, device=model.device, dtype=torch.bfloat16),
                    torch.empty(1, 0, device=model.device, dtype=torch.long),
                    torch.empty(1, 0, device=model.device, dtype=torch.bfloat16),
                )

            dist = Categorical(policy)
            action = dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze(0).detach())
            rewards.append(reward)
            dones.append(done)
            actions.append(action)
            
            state = next_state
            if done:
                break
        
        if done:
            values.append(torch.tensor([0.0], device=model.device, dtype=torch.bfloat16))
        else:
            policy, value = model(
                torch.tensor(next_state, device=model.device, dtype=torch.bfloat16).unsqueeze(0),
                torch.stack(states).unsqueeze(0),
                torch.tensor(actions, device=model.device, dtype=torch.long).unsqueeze(0),
                torch.tensor(rewards, device=model.device, dtype=torch.bfloat16).unsqueeze(0)
            )
            values.append(value.squeeze(0).detach())

    return torch.stack(states), torch.stack(actions), torch.stack(log_probs), torch.stack(values), torch.tensor(rewards, dtype=torch.bfloat16), dones

episode_rewards = []
p_losses = []
v_losses = []

for episode in tqdm(range(episodes)):
    states, actions, log_probs, values, rewards, dones = rollout(model, env)
    advantages, returns = compute_advantages(rewards, values, dones, gamma, lambda_gae)

    model.train()
    for _ in range(ppo_epochs):
        for batch_start in range(0, len(states), batch_size):
            batch_n = min(batch_size, len(states) - batch_start)
            batch_slice = slice(batch_start, batch_start + batch_n)
            batch_actions = actions[batch_slice]
            batch_log_probs = log_probs[batch_slice]
            batch_returns = returns[batch_slice].to(device=model.device)
            batch_advantages = advantages[batch_slice].to(device=model.device)

            batch_policies = []
            batch_values = []

            for i in range(batch_start, batch_start + batch_n):
                query_observation = states[i].unsqueeze(0)
                context_observations = states[:i].unsqueeze(0)
                context_actions = actions[:i].reshape(1,-1)
                context_rewards = rewards[:i].unsqueeze(0).to(device=model.device)

                policy, new_values = model(
                    query_observation,
                    context_observations,
                    context_actions,
                    context_rewards
                )
                batch_policies.append(policy)
                batch_values.append(new_values.squeeze())
            
            batch_policies = torch.stack(batch_policies)
            batch_values = torch.stack(batch_values)

            new_dist = Categorical(batch_policies)
            new_log_probs = new_dist.log_prob(batch_actions)
            ratio = (new_log_probs - batch_log_probs).exp()
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(batch_values, batch_returns)
            loss = policy_loss + 0.1 * value_loss
            p_losses.append(policy_loss.item())
            v_losses.append(value_loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ep_reward = rewards.sum().item()
    episode_rewards.append(ep_reward)
    print(f"Episode {episode}, Reward: {ep_reward}")

os.makedirs("plots", exist_ok=True)
plt.figure()
plt.plot(episode_rewards, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig(os.path.join("plots", "cartpole_rl2_rewards.png"))

plt.figure()
plt.plot(p_losses, label="Policy Loss")
plt.plot(v_losses, label="Value Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join("plots", "cartpole_rl2_losses.png"))

# TODO: Maybe instead of changing the input type, I can put state, action, reward as separate tokens and predict the action token given the state token and all preceding tuples
# TODO: figure it out why they couldn't reproduce DPT originally