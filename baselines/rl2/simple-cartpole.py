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

gamma = 0.99
lambda_gae = 0.95
eps_clip = 0.2
learning_rate = 3e-4
episodes = 1000
ppo_epochs = 5
batch_size = 64
seq_len = 4096

env = gym.make("CartPole-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, env.action_space.n),
    nn.Softmax(dim=-1)
).to(torch.float32)

critic = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(torch.float32)

policy_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)


def compute_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

def rollout(actor, critic, env):
    actor.eval()
    critic.eval()
    state, info = env.reset()
    log_probs = []
    values = []
    rewards = []
    dones = []
    states = []
    actions = []

    with torch.no_grad():
        for t in range(seq_len):
            state = torch.tensor(state, dtype=torch.float32)
            policy = actor(
                state.unsqueeze(0)
            )
            value = critic(
                state.unsqueeze(0)
            )

            dist = Categorical(policy)
            action = dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            log_probs.append(dist.log_prob(action).detach())
            values.append(value.squeeze(0).detach())
            rewards.append(reward)
            dones.append(done)
            actions.append(action)
            
            state = next_state
            if done:
                break
        
        if done:
            values.append(torch.tensor([0.0], dtype=torch.float32))
        else:
            value = critic(
                torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            )
            values.append(value.squeeze(0).detach())

    return torch.stack(states), torch.stack(actions), torch.stack(log_probs), torch.stack(values), torch.tensor(rewards, dtype=torch.float32), dones

def plot_rewards_losses(episode_rewards, p_losses, v_losses, advantages, policy_grads, value_grads):
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(episode_rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join("plots", "cartpole_rl2_rewards.png"))
    plt.close()

    plt.figure()
    plt.plot(p_losses, label="Policy Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "cartpole_rl2_policy_losses.png"))
    plt.close()

    plt.figure()
    plt.plot(v_losses, label="Value Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "cartpole_rl2_value_losses.png"))
    plt.close()

    plt.figure()
    plt.plot(advantages, label="Mean Advantage")
    plt.xlabel("Batch")
    plt.ylabel("Advantage")
    plt.legend()
    plt.savefig(os.path.join("plots", "cartpole_rl2_advantages.png"))
    plt.close()

    plt.figure()
    plt.plot(policy_grads, label="Policy Gradient Norm")
    plt.xlabel("Batch")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.savefig(os.path.join("plots", "cartpole_rl2_policy_gradients.png"))
    plt.close()

    plt.figure()
    plt.plot(value_grads, label="Value Gradient Norm")
    plt.xlabel("Batch")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.savefig(os.path.join("plots", "cartpole_rl2_value_gradients.png"))
    plt.close()

mean_policy_gradients = []
mean_value_gradients = []
episode_rewards = []
p_losses = []
v_losses = []
mean_advantages = []

# for episode in tqdm(range(episodes)):
for episode in range(episodes):
    states, actions, log_probs, values, rewards, dones = rollout(actor, critic, env)
    advantages, returns = compute_advantages(rewards, values, dones, gamma, lambda_gae)

    actor.train()
    critic.train()
    for _ in range(ppo_epochs):
        for batch_start in range(0, len(states), batch_size):
            batch_n = min(batch_size, len(states) - batch_start)
            batch_slice = slice(batch_start, batch_start + batch_n)
            batch_actions = actions[batch_slice]
            batch_log_probs = log_probs[batch_slice]
            batch_returns = returns[batch_slice]
            batch_advantages = advantages[batch_slice]

            mean_advantages.append(batch_advantages.mean().item())
            # Normalize advantages
            batch_advantages = batch_advantages - batch_advantages.mean()
            batch_advantages = batch_advantages / (batch_advantages.std() + 1e-8)

            # for i in range(batch_start, batch_start + batch_n):
            #     query_observation = states[i].unsqueeze(0)
                
            #     policy = model(
            #         query_observation
            #     )
            #     batch_policies.append(policy)
            batch_observations = states[batch_slice]
            
            batch_policies = actor(
                batch_observations
            )
            batch_values = critic(
                batch_observations
            ).squeeze(1)

            new_dist = Categorical(batch_policies)
            new_log_probs = new_dist.log_prob(batch_actions)
            ratio = (new_log_probs - batch_log_probs).exp()

            # print(f'{ratio=}')
            # print(f'{batch_advantages=}')
            # print(f'{batch_returns=}')
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            # print(surr1)
            # print(f'{policy_loss=}')

            policy_optimizer.zero_grad()
            policy_loss.backward()
            # Log policy gradient norms
            pg_norms = [p.grad.data.norm(2).item() for p in actor.parameters() if p.grad is not None]
            mean_policy_gradients.append(sum(pg_norms)/len(pg_norms) if pg_norms else 0.0)
            policy_optimizer.step()

            value_loss = F.mse_loss(batch_values, batch_returns)
            p_losses.append(policy_loss.item())
            v_losses.append(value_loss.item())
            
            value_optimizer.zero_grad()
            value_loss.backward()
            # Log value gradient norms
            vg_norms = [p.grad.data.norm(2).item() for p in critic.parameters() if p.grad is not None]
            mean_value_gradients.append(sum(vg_norms)/len(vg_norms) if vg_norms else 0.0)
            value_optimizer.step()


    ep_reward = rewards.sum().item()
    episode_rewards.append(ep_reward)
    print(f"Episode {episode}, Reward: {ep_reward}")
    if episode % 100 == 99:
        plot_rewards_losses(episode_rewards, p_losses, v_losses, mean_advantages, mean_policy_gradients, mean_value_gradients)


# class CartpolePolicy(nn.Module):
#     def __init__(
#             self,
#             num_actions: int,
#             observation_dim: int,
#             embedding_dim: int = 64,
#             num_layers: int = 4,
#             embedding_dropout: float = 0.1,
#             ) -> None:
#         super().__init__()

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.embedding_dim = embedding_dim
#         self.num_actions = num_actions

#         self.emb_dropout = nn.Dropout(embedding_dropout)

#         self.obs_encoder = nn.Linear(
#             observation_dim, embedding_dim
#         )
        
#         self.blocks = nn.ModuleList(
#             [
#                 nn.Linear(embedding_dim, embedding_dim)
#                 for _ in range(num_layers)
#             ]
#         )
        
#         self.action_head = nn.Linear(embedding_dim, num_actions)


#         self.apply(self._init_weights)

#         self.to(self.device)

#     @staticmethod
#     def _init_weights(module: nn.Module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             torch.nn.init.zeros_(module.bias)
#             torch.nn.init.ones_(module.weight)
#         if isinstance(module, nn.Conv2d):
#             gain = nn.init.calculate_gain("relu")
#             nn.init.orthogonal_(module.weight.data, gain)
#             if hasattr(module.bias, "data"):
#                 module.bias.data.fill_(0.0)
    
#     def forward(self,
#                 query_observations: torch.Tensor, # [batch_size, 4] or [batch_size, 4]
#                 ) -> torch.Tensor:

#         out = self.obs_encoder(query_observations)

#         for block in self.blocks:
#             out = F.relu(out + block(out))

#         logits = self.action_head(out)
#         policy = F.softmax(logits, dim=-1)
        
#         return policy



# class CartpoleValues(nn.Module):
#     def __init__(
#             self,
#             observation_dim: int,
#             embedding_dim: int = 64,
#             num_layers: int = 4,
#             embedding_dropout: float = 0.1,
#             ) -> None:
#         super().__init__()

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.embedding_dim = embedding_dim

#         self.emb_dropout = nn.Dropout(embedding_dropout)

#         self.obs_encoder = nn.Linear(
#             observation_dim, embedding_dim
#         )
        
#         self.blocks = nn.ModuleList(
#             [
#                 nn.Linear(embedding_dim, embedding_dim)
#                 for _ in range(num_layers)
#             ]
#         )
        
#         self.value_head = nn.Linear(embedding_dim, 1)

#         self.apply(self._init_weights)

#         self.to(self.device)

#     @staticmethod
#     def _init_weights(module: nn.Module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             torch.nn.init.zeros_(module.bias)
#             torch.nn.init.ones_(module.weight)
#         if isinstance(module, nn.Conv2d):
#             gain = nn.init.calculate_gain("relu")
#             nn.init.orthogonal_(module.weight.data, gain)
#             if hasattr(module.bias, "data"):
#                 module.bias.data.fill_(0.0)
    
#     def forward(self,
#                 query_observations: torch.Tensor, # [batch_size, 4] or [batch_size, 4]
#                 ) -> torch.Tensor:

#         out = self.obs_encoder(query_observations)

#         out = F.relu(out)

#         for block in self.blocks:
#             out = F.relu(out + block(out))

#         values = self.value_head(out)

#         return values

# actor = CartpolePolicy(
#     num_actions=env.action_space.n,
#     observation_dim=env.observation_space.shape[0],
#     embedding_dim=64,
#     num_layers=2,
# ).to(torch.bfloat16)

# critic = CartpoleValues(
#     observation_dim=env.observation_space.shape[0],
#     embedding_dim=64,
#     num_layers=2,
# ).to(torch.bfloat16)

