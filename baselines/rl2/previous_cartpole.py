import gym
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from torch.distributions import Categorical
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math

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
                query_observations: torch.Tensor, # [batch_size, 5, 5, 2] or [batch_size, 2, 5, 5]
                context_observations: torch.Tensor, # [batch_size, seq_len, 5, 5, 2] or [batch_size, seq_len, 2, 5, 5]
                context_actions: torch.Tensor, # [batch_size, seq_len]
                # context_next_observations: torch.Tensor, # [batch_size, seq_len, 5, 5, 2] or [batch_size, seq_len, 2, 5, 5]
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
            
            # [batch_size, seq_len + 1, embedding_dim]
            observation_seq = torch.cat([query_obs_emb, context_obs_emb], dim=1)

            action_seq = torch.cat(
                [zeros.to(context_actions.dtype).unsqueeze(-1).repeat_interleave(self.num_actions, dim=-1),
                context_actions_emb],
                dim=1
            )

            # [batch_size, seq_len + 1]
            reward_seq = torch.cat(
                [zeros.to(context_rewards.dtype),
                context_rewards],
                dim=1
            ).unsqueeze(-1)

            # TODO: I believe this is zeroing out the embeddings of rewards and actions for the query observation.
            # TODO: why is this adding the zeros to the first column?

            sequence = torch.cat(
                [observation_seq, action_seq, reward_seq], dim=-1
            )  # [batch_size, seq_len + 1, state_embedding_dim + num_actions + 1]

        sequence = self.embed_transition(sequence)

        out = self.emb_dropout(sequence)
        print('new')
        print(out.shape)
        for block in self.blocks:
            out = block(out)
            print(out.shape)

        # print(sequence.shape)
        # tgt = torch.zeros(batch_size, 0, self.hidden_dim, device=sequence.device, dtype=sequence.dtype)
        # out = self.transformer(sequence, tgt)
        # print(out.shape)
        
        last_out = out[:, -1, :]
        # [batch_size, seq_len + 1, num_actions]
        print(last_out.shape)
        logits = self.action_head(last_out)
        print(logits.shape)

        # [batch_size, seq_len + 1, 1]
        values = self.value_head(last_out)
        print(values.shape)
        print(values)

        policy = F.softmax(logits, dim=-1)
        return policy, values

        # if not self.training:
        #     return F.softmax(logits[:, -1, :]), value[:, -1, :]
        # # return logits[:, 1:, :]
        
        # policy = F.softmax(logits, dim=-1)

        # # [batch_size, seq_len + 1, num_actions], [batch_size, seq_len + 1, num_actions]
        # return policy, value


gamma = 0.99
lambda_gae = 0.95
eps_clip = 0.2
learning_rate = 3e-4
ppo_epochs = 10
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
).half()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def compute_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

def rollout(model, env):
    model.eval()
    state, info = env.reset()
    # cache = model.init_cache(batch_size=1, dtype=torch.float16, device="cpu")
    log_probs = []
    values = []
    rewards = []
    dones = []
    states = []
    actions = []

    for t in range(seq_len):
        state = torch.tensor(state, device=model.device, dtype=torch.float16)
        policy, value = model(
            state.unsqueeze(0),
            torch.stack(states).unsqueeze(0) if states else torch.Tensor(),
            torch.tensor(actions, device=model.device, dtype=torch.long).unsqueeze(0) if actions else torch.Tensor(),
            torch.tensor(rewards, device=model.device, dtype=torch.float16).unsqueeze(0) if rewards else torch.Tensor(),
            # torch.tensor(actions).unsqueeze(0) if actions else torch.zeros(1, 1, dtype=torch.int32),
            # torch.tensor(rewards).unsqueeze(0) if rewards else torch.zeros(1, 1),
        )
        dist = Categorical(policy.squeeze(0))
        action = dist.sample()
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        states.append(state)
        log_probs.append(dist.log_prob(action))
        values.append(value.squeeze(0))
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
        
        state = next_state
        if done:
            break
    
    if done:
        values.append(torch.tensor([0.0], device=model.device, dtype=torch.float16))
    else:
        policy, value = model(
            torch.tensor(next_state, device=model.device, dtype=torch.float32).unsqueeze(0),
            torch.stack(states).unsqueeze(0),
            torch.tensor(actions, device=model.device, dtype=torch.long).unsqueeze(0),
            torch.tensor(rewards, device=model.device, dtype=torch.long).unsqueeze(0)
        )
        values.append(value.squeeze(0))

    return states, actions, log_probs, values, rewards, dones

for episode in range(1000):
    states, actions, log_probs, values, rewards, dones = rollout(model, env)
    advantages, returns = compute_advantages(rewards, values, dones, gamma, lambda_gae)

    # TODO: fix below
    model.train()
    for _ in range(ppo_epochs):
        for i in range(0, len(states), batch_size):
            batch_slice = slice(i, i + batch_size)
            batch_states = torch.cat(states[batch_slice])
            batch_actions = torch.stack(actions[batch_slice])
            batch_log_probs = torch.stack(log_probs[batch_slice])
            batch_returns = returns[batch_slice]
            batch_advantages = advantages[batch_slice]
            
            policy, new_values = model(
                batch_states,
                torch.tensor(actions).unsqueeze(0),
                torch.tensor(rewards).unsqueeze(0)
            )
            new_dist = Categorical(policy.squeeze(0))
            new_log_probs = new_dist.log_prob(batch_actions)
            ratio = (new_log_probs - batch_log_probs).exp()
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values.squeeze(0), batch_returns)
            loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Episode {episode}, Reward: {sum(rewards)}")

# TODO: Maybe instead of changing the input type, I can put state, action, reward as separate tokens and predict the action token given the state token and all preceding tuples
# TODO: I could try getting rid of the next_state_observation
# TODO: why is the last time step appended to the front of the sequence?
# TODO: figure it out why they couldn't reproduce DPT originally