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

from src.model_tuples_cache import Transformer, KVCache
from src.nn import TransformerBlock, ObservationEncoder

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

neural_net = CartpoleRL2(
    num_actions=env.action_space.n,
    observation_dim=env.observation_space.shape[0],
    embedding_dim=128,
).bfloat16()
# neural_net = CartpoleMLP(
#     num_actions=env.action_space.n,
#     observation_dim=env.observation_space.shape[0],
#     embedding_dim=128,
# ).bfloat16()

optimizer = optim.Adam(neural_net.parameters(), lr=lr)

Transition = namedtuple('Transition',
                        ['obs', 'act', 'rew', 'next_obs', 'terminated', 'logits'])

def act(params, state, states, actions, next_states, rewards):
    '''
    Given a state and previous transitions, return the action and the logits.
    '''

    query_observations = state.unsqueeze(0)
    if states:
        context_observations = torch.stack(states).unsqueeze(0)
        context_actions = torch.tensor(actions, device=params.device, dtype=torch.long).unsqueeze(0)
        context_next_observations = torch.tensor(np.stack(next_states), device=params.device, dtype=torch.bfloat16).unsqueeze(0)
        context_rewards = torch.tensor(rewards, device=params.device, dtype=torch.bfloat16).unsqueeze(0)
    else:
        context_observations = torch.empty(1, 0, *env.observation_space.shape, device=params.device, dtype=torch.bfloat16)
        context_actions = torch.empty(1, 0, device=params.device, dtype=torch.long)
        context_next_observations = torch.empty(1, 0, *env.observation_space.shape, device=params.device, dtype=torch.bfloat16) 
        context_rewards = torch.empty(1, 0, device=params.device, dtype=torch.bfloat16)


    # obs = torch.tensor(obs, dtype=torch.bfloat16, device=params.device).unsqueeze(0)
    # logits, _ = params(state.unsqueeze(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    logits, _ = params(query_observations, context_observations, context_actions, context_next_observations, context_rewards)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    return action.item(), logits.detach()


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
    next_states = []
    next_values = []

    with torch.no_grad():
        for t in range(T):
            state = torch.tensor(state, device=model.device, dtype=torch.bfloat16)
            # action, logit = act(model, state, states, actions, next_states, rewards)

            if states:
                context_observations = torch.stack(states).unsqueeze(0).to(device=model.device)#, dtype=torch.bfloat16)
                context_actions = torch.tensor(actions, device=model.device, dtype=torch.long).reshape(1,-1)
                context_next_observations = torch.tensor(next_states, device=model.device, dtype=torch.bfloat16).unsqueeze(0)
                context_rewards = torch.tensor(rewards, device=model.device, dtype=torch.bfloat16).unsqueeze(0)
            else:
                context_observations = torch.empty(1, 0, *env.observation_space.shape, device=model.device, dtype=torch.bfloat16)
                context_actions = torch.empty(1, 0, device=model.device, dtype=torch.long)
                context_next_observations = torch.empty(1, 0, *env.observation_space.shape, device=model.device, dtype=torch.bfloat16)
                context_rewards = torch.empty(1, 0, device=model.device, dtype=torch.bfloat16)
            
            logit, value = model(
                state.unsqueeze(0),
                context_observations,
                context_actions,
                context_next_observations,
                context_rewards
            )

            action = torch.distributions.Categorical(logits=logit).sample().item()

            next_state, reward, terminated, truncated, info = env.step(action)#.item())

            states.append(state)
            next_states.append(next_state)
            logits.append(logit.detach().squeeze())
            # values.append(value.squeeze(0).detach())
            values.append(value.item())
            rewards.append(reward)
            terminated_flags.append(terminated)
            actions.append(action)

            # TODO: include current state as context?
            _, next_value = model(
                torch.tensor(next_state, device=model.device, dtype=torch.bfloat16).unsqueeze(0),
                context_observations,
                context_actions,
                context_next_observations,
                context_rewards
            )
            next_values.append(next_value.item())
            
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

    return torch.stack(states), torch.tensor(np.stack(next_states), device=model.device, dtype=torch.bfloat16), torch.tensor(actions), torch.stack(logits), torch.tensor(rewards, dtype=torch.bfloat16), torch.tensor(terminated_flags, dtype=torch.int32), torch.tensor(values), torch.tensor(next_values)

# def advantage_estimation(params, obs, next_obs, actions, rewards, terminated, gamma):
#     '''
#     Given a trajectory, estimate the advantages using truncated GAE.
#     '''
#     with torch.no_grad():
#         # TODO: add context here
#         _, values = params(obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
#         _, next_values = params(next_obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
#         # next_values = values[1:].clone().squeeze().to('cpu')
#         # values = values[:-1].clone().squeeze().to('cpu')
#     values, next_values = values.squeeze(), next_values.squeeze()
#     deltas = rewards + gamma * next_values * (1 - terminated) - values
#     advantages = torch.zeros_like(rewards)
#     advantages[-1] = deltas[-1]
#     for t in reversed(range(len(rewards) - 1)):
#         advantages[t] = deltas[t] + gamma * lmda * (1 - terminated[t]) * advantages[t + 1]
#     return advantages

def advantage_estimation(values, next_values, rewards, terminated):
    '''
    Given a trajectory, estimate the advantages using truncated GAE.
    '''
    deltas = rewards + gamma * next_values * (1 - terminated) - values
    advantages = torch.zeros_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(len(rewards) - 1)):
        advantages[t] = deltas[t] + gamma * lmda * (1 - terminated[t]) * advantages[t + 1]
    return advantages


def policy_loss(logits, actions, old_logits, advantages):
    log_probs = nn.functional.log_softmax(logits, dim=1)
    old_log_probs = nn.functional.log_softmax(old_logits.reshape(-1,2), dim=1)
    log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    old_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    return -torch.mean(surrogate)

# def value_loss(params, values, next_obs, rewards, terminated):
def value_loss(values, next_values, rewards, terminated):
    # _, next_values = params(next_obs, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    tde = rewards + gamma * next_values * (1 - terminated) - values
    return torch.mean(tde ** 2)

def ppo_loss(logits, old_logits, values, next_values, actions, advantages, rewards, terminated, loss_ratio=0.5):
    p_loss = policy_loss(logits, actions, old_logits, advantages)
    v_loss = value_loss(values, next_values, rewards, terminated)
    return p_loss + loss_ratio * v_loss

def evaluate(params, num_eps=10):
    env_eval = wrappers.RecordEpisodeStatistics(env)
    cum_rews = []
    states = []
    actions = []
    next_states = []
    rewards = []
    for _ in range(num_eps):
        state, _ = env_eval.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            state = torch.tensor(state, dtype=torch.bfloat16, device=params.device)
            # action, _ = act(params, state)
            action, _ = act(params, state, states, actions, next_states, rewards)
            states.append(state)
            actions.append(action)
            state, reward, terminated, truncated, info = env_eval.step(action)
            next_states.append(state)
            rewards.append(reward)
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
    states, next_states, actions, logits, rewards, terminated, values, next_values = unroll_policy(neural_net, T=unroll_size)
    # advantages = advantage_estimation(neural_net, states, next_states, actions, rewards, terminated, gamma)
    advantages = advantage_estimation(values, next_values, rewards, terminated)

    neural_net.train()
    for _ in range(ppo_epochs):
        for batch_start in range(0, len(states) - 1, mini_size):
            batch_n = min(mini_size, len(states) - 1 - batch_start)
            batch_slice = slice(batch_start, batch_start + batch_n)
            batch_actions = actions[batch_slice].to(device=neural_net.device)
            batch_terminated = terminated[batch_slice].to(device=neural_net.device)
            old_logits = logits[batch_slice]
            batch_rewards = rewards[batch_slice].to(device=neural_net.device)
            batch_advantages = advantages[batch_slice].to(device=neural_net.device)
            batch_obs = states[batch_slice].to(device=neural_net.device)
            batch_next_obs = next_states[batch_slice].to(device=neural_net.device)

            # batch_advantages = F.normalize(batch_advantages)

            # batch_logits, batch_values = neural_net(
            #     states[batch_slice],
            #     torch.empty(0),
            #     torch.empty(0),
            #     torch.empty(0),
            #     torch.empty(0)
            # )

            batch_logits = []
            batch_values = []
            batch_next_values = []

            for i in range(batch_start, batch_start + batch_n):
                query_observation = states[i].unsqueeze(0)
                context_observations = states[:i].unsqueeze(0)
                context_actions = actions[:i].reshape(1,-1).to(device=neural_net.device)
                context_next_observations = next_states[:i].unsqueeze(0)
                context_rewards = rewards[:i].unsqueeze(0).to(device=neural_net.device)
                
                logs, values = neural_net(
                    query_observation,
                    context_observations,
                    context_actions,
                    context_next_observations,
                    context_rewards
                )
                batch_logits.append(logs.squeeze())
                batch_values.append(values.squeeze())
                # print(values.squeeze().grad)

                query_next_observation = next_states[i].unsqueeze(0)
                # context_observations = states[:i+1].unsqueeze(0)
                # context_actions = actions[:i+1].reshape(1,-1)
                # context_next_observations = next_states[:i+1].unsqueeze(0)
                # context_rewards = rewards[:i+1].unsqueeze(0).to(device=neural_net.device)

                _, next_values = neural_net(
                    query_next_observation,
                    context_observations,
                    context_actions,
                    context_next_observations,
                    context_rewards
                )
                batch_next_values.append(next_values.squeeze().detach())
            
            batch_logits = torch.stack(batch_logits)
            batch_values = torch.stack(batch_values)
            batch_next_values = torch.stack(batch_next_values)

            optimizer.zero_grad()
            loss = ppo_loss(batch_logits, old_logits, batch_values, batch_next_values, batch_actions, batch_advantages, batch_rewards, batch_terminated, loss_ratio=0.5)
            loss.backward()
            optimizer.step()

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

state, _ = test_env.reset(seed=42)
terminated, truncated = False, False
states = []
actions = []
next_states = []
rewards = []
while not terminated and not truncated:
    state = torch.tensor(state, dtype=torch.bfloat16, device=neural_net.device)
    action, _ = act(neural_net, state, states, actions, next_states, rewards)
    states.append(state)
    actions.append(action)
    state, reward, terminated, truncated, info = test_env.step(action)
    next_states.append(state)
    rewards.append(reward)
test_env.close()


            # batch_logits = []
            # batch_values = []

            # for i in range(batch_start, batch_start + batch_n):
            #     query_observation = states[i].unsqueeze(0)
            #     # context_observations = states[:i].unsqueeze(0)
            #     # context_actions = actions[:i].reshape(1,-1)
            #     # context_rewards = rewards[:i].unsqueeze(0).to(device=model.device)
            #     context_observations = torch.empty(0)
            #     context_actions = torch.empty(0)
            #     context_rewards = torch.empty(0)
            #     context_next_observations = torch.empty(0)
                
            #     policy, new_values = neural_net(
            #         query_observation,
            #         context_observations,
            #         context_actions,
            #         context_next_observations,
            #         context_rewards
            #     )
            #     batch_logits.append(policy.squeeze())
            #     batch_values.append(new_values.squeeze())
            
            
            # batch_logits = torch.stack(batch_logits)
            # batch_values = torch.stack(batch_values)



            # query_observation = states[batch_start + batch_n].unsqueeze(0)
            # context_observations = torch.empty(0)
            # context_actions = torch.empty(0)
            # context_rewards = torch.empty(0)
            # context_next_observations = torch.empty(0)
            
            # _, new_values = neural_net(
            #     query_observation,
            #     context_observations,
            #     context_actions,
            #     context_next_observations,
            #     context_rewards
            # )
            # batch_values.append(new_values.squeeze())
