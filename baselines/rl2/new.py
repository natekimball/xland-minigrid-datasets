import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from collections import namedtuple
from gymnasium import wrappers
import math

# from src.model_tuples_cache import Transformer, KVCache
# from src.nn import TransformerBlock, ObservationEncoder

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

        self.emb_dropout = nn.Dropout(embedding_dropout)

        self.obs_encoder = nn.Linear(
            observation_dim, embedding_dim
        )
        
        self.embed_transition = nn.Linear(
            2 * embedding_dim + num_actions + 1, # [state, next_state, action, reward]
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
                context_next_observations: torch.Tensor, # [batch_size, seq_len, 4] or [batch_size, seq_len, 4]
                context_rewards: torch.Tensor, # [batch_size, seq_len]
                ) -> torch.Tensor:
        # TODO: If I can't do positional encoding, I need to encode entire transition with next state
        # TODO: investigate AD to see how they handle positional encoding
        if context_observations.numel() == 0:
            batch_size, seq_len = query_observations.shape[0], 0
            # [batch_size, 1, embedding_dim]
            query_obs_emb = self.obs_encoder(query_observations.unsqueeze(1))

            zeros = torch.zeros(batch_size, 1, self.embedding_dim + self.num_actions + 1, device=query_observations.device, dtype=query_observations.dtype)

            sequence = torch.cat(
                [query_obs_emb, zeros], dim=-1
            )  # [batch_size, seq_len + 1, state_embedding_dim + num_actions + 1]

        else:
            batch_size, seq_len = context_rewards.shape[0], context_rewards.shape[1]
            
            # [batch_size, 1, embedding_dim]
            query_obs_emb = self.obs_encoder(query_observations.unsqueeze(1))
            
            # [batch_size, seq_len, embedding_dim]
            context_obs_emb = self.obs_encoder(context_observations)

            # [batch_size, seq_len, embedding_dim]
            context_next_obs_emb = self.obs_encoder(context_next_observations)

            # [batch_size, seq_len, num_actions]
            context_actions_emb = F.one_hot(context_actions, num_classes=self.num_actions)

            zeros = torch.zeros(batch_size, 1, device=context_observations.device, dtype=context_observations.dtype)

            # [batch_size, seq_len + 1, embedding_dim]
            observation_seq = torch.cat([context_obs_emb, query_obs_emb], dim=1)

            action_seq = torch.cat(
                [context_actions_emb,
                zeros.to(context_actions.dtype).unsqueeze(-1).repeat_interleave(self.num_actions, dim=-1)],
                dim=1
            )

            next_observation_seq = torch.cat(
                [context_next_obs_emb,
                zeros.unsqueeze(-1).repeat_interleave(self.embedding_dim, dim=-1)],
                dim=1
            )

            # [batch_size, seq_len + 1, 1]
            reward_seq = torch.cat(
                [context_rewards,
                zeros.to(context_rewards.dtype)],
                dim=1
            ).unsqueeze(-1)

            sequence = torch.cat(
                [observation_seq, action_seq, next_observation_seq, reward_seq], dim=-1
            )  # [batch_size, seq_len + 1, 2 * state_embedding_dim + num_actions + 1]


        sequence = self.embed_transition(sequence)
        sequence = self.pos_encoder(sequence)

        out = self.emb_dropout(sequence)
        for block in self.blocks:
            out = block(out)
        
        last_out = out[:, -1, :]
        # [batch_size, 1]
        values = self.value_head(last_out)

        # [batch_size, num_actions]
        logits = self.action_head(last_out)
        # policy = F.softmax(logits, dim=-1)

        return logits, values

class CartpoleMLP(nn.Module):
    def __init__(
            self,
            num_actions: int,
            observation_dim: int,
            embedding_dim: int = 64,
            num_layers: int = 4,
            embedding_dropout: float = 0.1,
            ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions

        self.emb_dropout = nn.Dropout(embedding_dropout)

        self.obs_encoder = nn.Linear(
            observation_dim, embedding_dim
        )
        
        self.blocks = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim)
                for _ in range(num_layers)
            ]
        )
        
        self.action_layer = nn.Linear(embedding_dim, embedding_dim)
        self.action_head = nn.Linear(embedding_dim, num_actions)

        self.value_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_head = nn.Linear(embedding_dim, 1)

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
                context_next_observations: torch.Tensor, # [batch_size, seq_len, 4] or [batch_size, seq_len, 4]
                context_rewards: torch.Tensor, # [batch_size, seq_len]
                ) -> torch.Tensor:
        # TODO: If I can't do positional encoding, I need to encode entire transition with next state
        # TODO: investigate AD to see how they handle positional encoding

        out = self.obs_encoder(query_observations)

        for block in self.blocks:
            out = out + block(out)

        value_out = self.value_layer(out)
        values = self.value_head(value_out)

        action_out = self.action_layer(out)
        logits = self.action_head(action_out)
        # policy = F.softmax(logits, dim=-1)
        
        return logits, values

gamma = 0.99
lmda = 0.95
epsilon = 0.2
seed = 0
lr = 0.001
ppo_epochs = 4

env = gym.make('CartPole-v1')
print('observation shape:', env.observation_space.shape)
print('action shape:', env.action_space.shape)

# neural_net = CartpoleRL2(
#     num_actions=env.action_space.n,
#     observation_dim=env.observation_space.shape[0],
#     embedding_dim=128,
# ).bfloat16()
neural_net = CartpoleMLP(
    num_actions=env.action_space.n,
    observation_dim=env.observation_space.shape[0],
    embedding_dim=128,
).bfloat16()

optimizer = optim.Adam(neural_net.parameters(), lr=lr)

Transition = namedtuple('Transition',
                        ['obs', 'act', 'rew', 'next_obs', 'terminated', 'logits'])

def act(params, obs):
    '''
    Given the current state, return the action and the logits.
    '''
    # obs = torch.tensor(obs, dtype=torch.bfloat16, device=params.device).unsqueeze(0)
    logits, _ = params(obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    return action.item(), logits.detach()

# def unroll_policy(model, T=200):
#     model.eval()
#     state, info = env.reset()
#     # cache = model.init_cache(batch_size=1, dtype=torch.bfloat16, device="cpu")
#     log_probs = []
#     values = []
#     rewards = []
#     terminated_flags = []
#     states = []
#     actions = []
#     logits = []
#     trajectory = []

#     with torch.no_grad():
#         for t in range(T):
#             # state = torch.tensor(state, device=model.device, dtype=torch.bfloat16)
#             # if states:
#             #     policy, value = model(
#             #         state.unsqueeze(0),
#             #         torch.stack(states).unsqueeze(0),
#             #         torch.tensor(actions, device=model.device, dtype=torch.long).unsqueeze(0),
#             #         torch.tensor(rewards, device=model.device, dtype=torch.bfloat16).unsqueeze(0)
#             #     )
#             # else:
#             #     policy, value = model(
#             #         state.unsqueeze(0),
#             #         torch.empty(1, 0, *env.observation_space.shape, device=model.device, dtype=torch.bfloat16),
#             #         torch.empty(1, 0, device=model.device, dtype=torch.long),
#             #         torch.empty(1, 0, device=model.device, dtype=torch.bfloat16),
#             #     )
#             # policy, value = model(
#             #     state.unsqueeze(0),
#             #     torch.empty(1, 0, *env.observation_space.shape, device=model.device, dtype=torch.bfloat16),
#             #     torch.empty(1, 0, device=model.device, dtype=torch.long),
#             #     torch.empty(1, 0, device=model.device, dtype=torch.bfloat16),
#             # )


#             # dist = Categorical(policy)
#             # action = dist.sample()
#             action, logit = policy(model, state)
#             next_state, reward, terminated, truncated, info = env.step(action)#.item())

#             states.append(state)
#             # log_probs.append(dist.log_prob(action))
#             # log_probs.append(F.log_softmax(logits, dim=1).gather(1, action.unsqueeze(1)).squeeze())
#             logits.append(logit)
#             # values.append(value.squeeze(0).detach())
#             rewards.append(reward)
#             terminated_flags.append(terminated)
#             actions.append(action)
#             trajectory.append(Transition(state, action, reward, next_state, terminated, logit))
            
#             if terminated or truncated:
#                 state, _ = env.reset()
#             else:
#                 state = next_state
        
#     return trajectory
#     # return torch.stack(states), torch.stack(actions), torch.stack(log_probs), torch.stack(values), torch.tensor(rewards, dtype=torch.bfloat16), terminated_flags

def unroll_policy(model, T=200):
    model.eval()
    state, _ = env.reset()
    # cache = model.init_cache(batch_size=1, dtype=torch.bfloat16, device="cpu")
    log_probs = []
    values = []
    rewards = []
    terminated_flags = []
    states = []
    actions = []
    logits = []
    trajectory = []

    with torch.no_grad():
        for t in range(T):
            state = torch.tensor(state, device=model.device, dtype=torch.bfloat16)

            # dist = Categorical(policy)
            # action = dist.sample()
            action, logit = act(model, state)
            next_state, reward, terminated, truncated, info = env.step(action)#.item())

            states.append(state)
            # log_probs.append(dist.log_prob(action))
            # log_probs.append(F.log_softmax(logits, dim=1).gather(1, action.unsqueeze(1)).squeeze())
            logits.append(logit.detach().squeeze())
            # values.append(value.squeeze(0).detach())
            rewards.append(reward)
            terminated_flags.append(terminated)
            actions.append(action)
            trajectory.append(Transition(state, action, reward, next_state, terminated, logit))
            
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

    states.append(torch.tensor(next_state, device=model.device, dtype=torch.bfloat16)) 

    return trajectory
    # return torch.stack(states), torch.tensor(actions), torch.stack(logits), torch.tensor(rewards, dtype=torch.bfloat16), torch.tensor(terminated_flags, dtype=torch.int32)


def advantage_estimation(trajectory, params):
    '''
    Given a trajectory, estimate the advantages using truncated GAE.
    '''
    obs = torch.stack([t.obs for t in trajectory]).bfloat16()
    next_obs = torch.tensor([t.next_obs for t in trajectory]).bfloat16()
    rewards = torch.tensor(np.array([t.rew for t in trajectory]), dtype=torch.bfloat16, device=params.device)
    terminated = torch.tensor(np.array([t.terminated for t in trajectory]), dtype=torch.bfloat16, device=params.device)
    
    _, values = params(obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    _, next_values = params(next_obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    values, next_values = values.squeeze(), next_values.squeeze()
    deltas = rewards + gamma * next_values * (1 - terminated) - values
    advantages = torch.zeros_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(len(trajectory) - 1)):
        advantages[t] = deltas[t] + gamma * lmda * (1 - terminated[t]) * advantages[t + 1]
    return advantages

# def advantage_estimation(params, rewards, values, terminated, gamma):
#     '''
#     Given a trajectory, estimate the advantages using truncated GAE.
#     '''
#     obs = torch.tensor(np.array([t.obs for t in trajectory]), dtype=torch.bfloat16, device=params.device)
#     next_obs = torch.tensor(np.array([t.next_obs for t in trajectory]), dtype=torch.bfloat16, device=params.device)
#     rewards = torch.tensor(np.array([t.rew for t in trajectory]), dtype=torch.bfloat16, device=params.device)
#     terminated = torch.tensor(np.array([t.terminated for t in trajectory]), dtype=torch.bfloat16, device=params.device)
    
#     _, values = params(obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
#     _, next_values = params(next_obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
#     values, next_values = values.squeeze(), next_values.squeeze()
#     deltas = rewards + gamma * next_values * (1 - terminated) - values
#     advantages = torch.zeros_like(rewards)
#     advantages[-1] = deltas[-1]
#     for t in reversed(range(len(rewards) - 1)):
#         advantages[t] = deltas[t] + gamma * lmda * (1 - terminated[t]) * advantages[t + 1]
#     return advantages

def policy_loss(params, obs, actions, old_logits, advantages):
    logits, _ = params(obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    log_probs = nn.functional.log_softmax(logits, dim=1)
    old_log_probs = nn.functional.log_softmax(old_logits.reshape(-1,2), dim=1)
    log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    old_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    return -torch.mean(surrogate)

def value_loss(params, obs, next_obs, rewards, terminated):
    _, values = params(obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    _, next_values = params(next_obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    values, next_values = values.squeeze(), next_values.squeeze()
    tde = rewards + gamma * next_values * (1 - terminated) - values
    return torch.mean(tde ** 2)

def ppo_loss(params, obs, actions, old_logits, advantages, next_obs, rewards, terminated, loss_ratio=0.5):
    p_loss = policy_loss(params, obs, actions, old_logits, advantages)
    v_loss = value_loss(params, obs, next_obs, rewards, terminated)
    return p_loss + loss_ratio * v_loss

def update(params, optimizer, traj_segment, advantage_segment):
    # obs = torch.tensor([t.obs for t in traj_segment], dtype=torch.bfloat16, device=params.device)
    obs = torch.stack([t.obs for t in traj_segment]).bfloat16()
    actions = torch.tensor([t.act for t in traj_segment], dtype=torch.int64, device=params.device)

    old_logits = torch.stack([t.logits for t in traj_segment])
    advantages = advantage_segment.detach()
    next_obs = torch.tensor([t.next_obs for t in traj_segment], dtype=torch.bfloat16, device=params.device)
    rewards = torch.tensor([t.rew for t in traj_segment], dtype=torch.bfloat16, device=params.device)
    terminated = torch.tensor([t.terminated for t in traj_segment], dtype=torch.int32, device=params.device)
    
    optimizer.zero_grad()
    loss = ppo_loss(params, obs, actions, old_logits, advantages, next_obs, rewards, terminated, loss_ratio=0.5)
    loss.backward()
    optimizer.step()

def evaluate(params, num_eps=10):
    env_eval = wrappers.RecordEpisodeStatistics(env)
    cum_rews = []
    for _ in range(num_eps):
        obs, _ = env_eval.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            obs = torch.tensor(obs, dtype=torch.bfloat16, device=params.device)
            action, _ = act(params, obs)
            obs, _, terminated, truncated, info = env_eval.step(action)
        cum_rews.append(info['episode']['r'])
    return np.mean(cum_rews)

steps = 150
unroll_size = 1280
mini_size = 128
eval_freq = 10
best_params = neural_net.state_dict()
best_mean_ret = 0
eval_steps = []
mean_returns = []

for step in range(1, steps + 1):
    traj = unroll_policy(neural_net, T=unroll_size)
    advantage = advantage_estimation(traj, neural_net)
#     # states, actions, log_probs, values, rewards, terminated_flags = unroll_policy(neural_net, env)
#     # advantages, returns = advantage_estimation(neural_net, rewards, values, terminated_flags, gamma)


#     neural_net.train()
#     for batch_start in range(0, len(traj), mini_size):
#         batch_n = min(mini_size, len(traj) - batch_start)
#         batch_slice = slice(batch_start, batch_start + batch_n)
#         batch_actions = actions[batch_slice]
#         batch_log_probs = log_probs[batch_slice]
#         batch_returns = returns[batch_slice].to(device=model.device)
#         batch_advantages = advantages[batch_slice].to(device=model.device)

#         # Normalize advantages
#         batch_advantages = batch_advantages - batch_advantages.mean()
#         batch_advantages = batch_advantages / (batch_advantages.std() + 1e-8)

#         batch_policies = []
#         batch_values = []

#         for i in range(batch_start, batch_start + batch_n):
#             query_observation = states[i].unsqueeze(0)
#             # context_observations = states[:i].unsqueeze(0)
#             # context_actions = actions[:i].reshape(1,-1)
#             # context_rewards = rewards[:i].unsqueeze(0).to(device=model.device)
#             context_observations = torch.empty(0)
#             context_actions = torch.empty(0)
#             context_rewards = torch.empty(0)
            
#             policy, new_values = model(
#                 query_observation,
#                 context_observations,
#                 context_actions,
#                 context_rewards
#             )
#             batch_policies.append(policy)
#             batch_values.append(new_values.squeeze())
        
#         batch_policies = torch.stack(batch_policies)
#         batch_values = torch.stack(batch_values)

#         new_dist = Categorical(batch_policies)
#         new_log_probs = new_dist.log_prob(batch_actions)
#         ratio = (new_log_probs - batch_log_probs).exp()
        
#         surr1 = ratio * batch_advantages
#         surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * batch_advantages
#         policy_loss = -torch.min(surr1, surr2).mean()
#         # print(surr1, surr2, policy_loss)
#         # value_loss = .00001 * F.mse_loss(batch_values, batch_returns)
#         # loss = policy_loss + value_loss
#         p_losses.append(policy_loss.item())
#         # v_losses.append(value_loss.item())
        
#         optimizer.zero_grad()
#         # loss.backward()
#         policy_loss.backward()
#         optimizer.step()

    for _ in range(ppo_epochs):
        # Shuffle data for better learning
        indices = np.arange(unroll_size)
        np.random.shuffle(indices)
        
        # Process mini-batches
        for start in range(0, unroll_size, mini_size):
            end = start + mini_size
            if end <= unroll_size:
                idx = indices[start:end]
                mini_traj = [traj[i] for i in idx]
                mini_adv = advantage[idx]
                update(neural_net, optimizer, mini_traj, mini_adv)

    # for _ in range(ppo_epochs):
    #     for i in range(0, unroll_size, mini_size):
    #         # TODO: add context
    #         # TODO: refactor to avoid Trajectory object
    #         # TODO: before refactoring Trajectory, get it working with context?
    #         mini_traj = traj[i:i + mini_size]
    #         mini_adv = advantage[i:i + mini_size]
    #         update(neural_net, optimizer, mini_traj, mini_adv)

    if step % eval_freq == 0:
        mean_ret = evaluate(neural_net)
        eval_steps.append(step)
        mean_returns.append(mean_ret)
        if mean_ret > best_mean_ret:
            best_mean_ret = mean_ret
            best_params = neural_net.state_dict()
        print(f'step: {step}, mean return: {mean_ret}')
        torch.save({'model_state_dict': best_params}, f'checkpoints/ckpt_{step}.pth')

torch.save({'model_state_dict': best_params}, 'checkpoints/ckpt_last.pth')
plt.figure()
plt.plot(eval_steps, mean_returns, label='mean returns')
plt.legend()
plt.savefig('mean_returns.png')
env.close()

# Generate video
test_env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=500)
test_env = wrappers.RecordVideo(env=test_env, video_folder='./video', name_prefix='PPO', disable_logger=True)

obs, _ = test_env.reset(seed=42)
terminated, truncated = False, False
while not terminated and not truncated:
    action, _ = act(neural_net, obs)
    obs, _, terminated, truncated, _ = test_env.step(action)
test_env.close()


# import os
# import pickle
# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from collections import namedtuple
# from gymnasium import wrappers
# from utils.common import KeyGenerator


# class NeuralNet(nn.Module):
#     def __init__(self, input_dim, hidden_size=128, action_space=2):
#         super(NeuralNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.logits = nn.Linear(hidden_size, action_space)
#         self.value = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         logits = self.logits(x)
#         value = self.value(x)
#         return logits, value

# gamma = 0.99
# lmda = 0.95
# epsilon = 0.2
# seed = 0
# lr = 0.0003
# n_epochs = 4
# value_coeff = 0.5
# entropy_coeff = 0.01

# torch.manual_seed(seed)

# env = gym.make('CartPole-v1')
# input_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# neural_net = NeuralNet(input_dim, action_space=action_dim)
# optimizer = optim.Adam(neural_net.parameters(), lr=lr)
# Transition = namedtuple('Transition', ['obs', 'act', 'rew', 'next_obs', 'terminated', 'logits'])
# obs, _ = env.reset(seed=seed)

# def policy(params, obs):
#     obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#     logits, _ = params(obs)
#     action = torch.distributions.Categorical(logits=logits).sample()
#     return action.item(), logits.detach().numpy()

# def unroll_policy(params, T=100):
#     trajectory = []
#     obs, _ = env.reset()
#     for _ in range(T):
#         action, logits = policy(params, obs)
#         next_obs, reward, terminated, truncated, _ = env.step(action)
#         trajectory.append(Transition(obs, action, reward, next_obs, terminated, logits))
#         obs = next_obs if not (terminated or truncated) else env.reset()[0]
#     return trajectory

# def compute_advantages(trajectory, params):
#     obs = torch.tensor([t.obs for t in trajectory], dtype=torch.float32)
#     next_obs = torch.tensor([t.next_obs for t in trajectory], dtype=torch.float32)
#     rewards = torch.tensor([t.rew for t in trajectory], dtype=torch.float32)
#     terminated = torch.tensor([t.terminated for t in trajectory], dtype=torch.float32)
    
#     with torch.no_grad():
#         _, values = params(obs)
#         _, next_values = params(next_obs)
#         values, next_values = values.squeeze(), next_values.squeeze()
        
#         deltas = rewards + gamma * next_values * (1 - terminated) - values
#         advantages = torch.zeros_like(rewards)
#         adv = 0
#         for t in reversed(range(len(trajectory))):
#             adv = deltas[t] + gamma * lmda * (1 - terminated[t]) * adv
#             advantages[t] = adv
        
#         # Normalize advantages - critical for stable training
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
#     returns = advantages + values  # For value function training
#     return advantages, returns

# def ppo_loss(params, batch, advantages, returns):
#     obs = torch.tensor(batch['obs'], dtype=torch.float32)
#     actions = torch.tensor(batch['actions'], dtype=torch.long)
#     old_logits = torch.tensor(batch['logits'], dtype=torch.float32)
    
#     logits, values = params(obs)
#     values = values.squeeze()
    
#     # Policy loss with proper log prob calculation
#     log_probs = nn.functional.log_softmax(logits, dim=1)
#     old_log_probs = nn.functional.log_softmax(old_logits.reshape(-1,2), dim=1)

#     log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
#     old_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    
#     ratio = torch.exp(log_probs - old_log_probs)
#     surrogate1 = ratio * advantages
#     surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
#     policy_loss = -torch.min(surrogate1, surrogate2).mean()
    
#     # Value loss
#     value_loss = value_coeff * ((values - returns) ** 2).mean()
    
#     # Entropy bonus for exploration
#     # entropy = dist.entropy().mean()
    
#     # Total loss
#     loss = policy_loss + value_loss #- entropy_coeff * entropy
    
#     return loss, policy_loss, value_loss#, entropy

# def update(params, optimizer, traj_segment, advantages, returns):
#     batch = {
#         'obs': np.array([t.obs for t in traj_segment]),
#         'actions': np.array([t.act for t in traj_segment]),
#         'logits': np.array([t.logits for t in traj_segment]),
#     }
    
#     # loss, policy_loss, value_loss, entropy = ppo_loss(params, batch, advantages, returns)
#     loss, policy_loss, value_loss = ppo_loss(params, batch, advantages, returns)
    
#     optimizer.zero_grad()
#     loss.backward()
#     # Add gradient clipping for stability
#     torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=0.5)
#     optimizer.step()
    
#     return loss.item(), policy_loss.item(), value_loss.item()#, entropy.item()

# def evaluate(params, num_eps=10):
#     env_eval = wrappers.RecordEpisodeStatistics(env)
#     cum_rews = []
#     for _ in range(num_eps):
#         obs, _ = env_eval.reset()
#         terminated, truncated = False, False
#         while not terminated and not truncated:
#             action, _ = policy(params, obs)
#             obs, _, terminated, truncated, info = env_eval.step(action)
#         cum_rews.append(info['episode']['r'])
#     return np.mean(cum_rews)

# steps = 150
# unroll_size = 1280
# mini_size = 128
# eval_freq = 10
# best_params = neural_net.state_dict()
# best_mean_ret = 0
# eval_steps, mean_returns = [], []

# for step in range(1, steps + 1):
#     traj = unroll_policy(neural_net, T=unroll_size)
#     advantages, returns = compute_advantages(traj, neural_net)
    
#     # Multiple optimization epochs over the same data
#     for _ in range(n_epochs):
#         # Shuffle data for better learning
#         indices = np.arange(unroll_size)
#         np.random.shuffle(indices)
        
#         # Process mini-batches
#         for start in range(0, unroll_size, mini_size):
#             end = start + mini_size
#             if end <= unroll_size:
#                 idx = indices[start:end]
#                 mini_traj = [traj[i] for i in idx]
#                 mini_adv = advantages[idx]
#                 mini_ret = returns[idx]
#                 update(neural_net, optimizer, mini_traj, mini_adv, mini_ret)

#     if step % eval_freq == 0:
#         mean_ret = evaluate(neural_net)
#         eval_steps.append(step)
#         mean_returns.append(mean_ret)
#         if mean_ret > best_mean_ret:
#             best_mean_ret = mean_ret
#             best_params = neural_net.state_dict()
#         print(f'step: {step}, mean return: {mean_ret}')
#         torch.save({'model_state_dict': best_params, 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoints/ckpt_{step}.pth')

# torch.save({'mel_state_dict': best_params, 'optimizer_state_dict': optimizer.state_dict()}, 'checkpoints/ckpt_last.pth')
# plt.figure()
# plt.plot(eval_steps, mean_returns, label='mean returns')
# plt.legend()
# plt.savefig('mean_returns.png')
# env.close()
