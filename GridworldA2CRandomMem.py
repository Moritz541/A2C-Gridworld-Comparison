import os
import random
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from GridworldAgentEnvRandomMem import GridWorldEnv, GridWorldA2C, count_parameters


# Experiment hyperparameters
USE_MEMORY = True
SEED = 0

# Learning hyperparameters
NUM_EPISODES = 30000
LEARNING_RATE = 0.0006
GAMMA = 0.97
REWARD_STEPS = 15
CLIP_GRADS = 0.15

# Evaluation hyperparameters
REWARD_WINDOW = 100
STOP_REWARD = 0.7

# Display hyperparameters
OUTPUT_EPISODES = 50

# Loss hyperparameters
VALUE_COEF = 0.4
ENTROPY_BETA_START =  0.02
ENTROPY_BETA_END = 0.0001

# Environment hyperparameters
NUM_ENVS = 16
MAX_STEPS = 50
WALL_PROB = 0.6
MIN_DIST = 4
VISIT_CAP = 5

# Set seeds so runs can be compared
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create writer
variant = "Mem" if USE_MEMORY else "noMem"
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(r"C:\Users\morit\Documents\A_Programmieren\DRL\A2C\Gridworld\runs", f"Random_gridworld_{variant}_{run_name}")
writer = SummaryWriter(log_dir=log_dir)

# Output device used
print("Using:", DEVICE)

# Create and initialize environments
envs = [GridWorldEnv(max_steps=MAX_STEPS, wall_prob = WALL_PROB, min_dist = MIN_DIST, use_memory = USE_MEMORY, visit_cap = VISIT_CAP) for _ in range(NUM_ENVS)]
states = np.array([env.reset() for env in envs], dtype=np.float32)

# Create agent and optimizer (the MLP needs no hidden state)
net = GridWorldA2C(input_shape=states[0].shape, n_actions=envs[0].n_actions).to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Output variant, observation size and number of parameters
print(f"Variant: {variant} | obs size: {states.shape[1]} | parameters: {count_parameters(net)}")

# Initialize episode tracking variables
reward_history = []
episode_step_history = []
finish_history = []
episode_rewards = np.zeros(NUM_ENVS)
episode_steps = np.zeros(NUM_ENVS)
episode_count = 0
global_step = 0
solved = False
mean_reward = -999.0

# Train until max episodes are reached or game is solved
while episode_count < NUM_EPISODES and not solved:

    # Initialize rollout tracking variables
    rollout_rewards = []
    rollout_log_probs = []
    rollout_values = []
    rollout_entropies = []
    rollout_dones = []

    # Take steps to fill up rollout
    for step in range(REWARD_STEPS):

        # Get current states and predictions
        state_t = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        logits, values = net(state_t)

        # Get masks from all environments (they are cached inside each environment)
        action_masks = torch.as_tensor(
            np.stack([env.get_action_mask() for env in envs]),
            dtype=torch.bool, device=DEVICE)

        # Mask invalid logits and sample actions from masked distribution
        masked_logits = logits.masked_fill(~action_masks, -1e9)
        distribution = torch.distributions.Categorical(logits=masked_logits)

        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        entropies = distribution.entropy()
        actions_np = actions.detach().cpu().numpy()


        # Initialize step tracking variables
        next_states = []
        rewards = []
        dones = []

        # Go through each environment to take step and check done
        for i, env in enumerate(envs):

            # Take step
            next_state, reward, done, finished = env.step(int(actions_np[i]))

            # Update step tracking variables
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            # Update episode tracking variables
            episode_rewards[i] += reward
            episode_steps[i] += 1

            # If done write into tensorboard and check if solved
            if done:

                # Update reward history with reward of total episodes
                finished_reward = episode_rewards[i]
                reward_history.append(finished_reward)
                finished_episode_steps = episode_steps[i]
                episode_step_history.append(finished_episode_steps)
                finish_history.append(finished)

                # Write into tensorboard
                writer.add_scalar("reward/episode", finished_reward, episode_count)
                writer.add_scalar("reward/average_100", np.mean(reward_history[-100:]), episode_count)
                writer.add_scalar("game/steps", finished_episode_steps, episode_count)
                writer.add_scalar("game/reached_goal", int(finished), episode_count)

                # Print episode information at certain threshold
                if episode_count % OUTPUT_EPISODES == 0 and episode_count != 0:
                    print(
                        f"Episode {episode_count:4d} | "
                        f"Mean reward last {OUTPUT_EPISODES} episodes: {np.mean(reward_history[-OUTPUT_EPISODES:]):6.3f} | "
                        f"Mean steps last {OUTPUT_EPISODES} episodes: {np.mean(episode_step_history[-OUTPUT_EPISODES:]):.3f} | "
                        f"# of finished episodes of the last {OUTPUT_EPISODES}: {np.sum(finish_history[-OUTPUT_EPISODES:]):.0f}"
                    )

                # Update/reset episode tracking variables
                episode_count += 1
                episode_rewards[i] = 0
                episode_steps[i] = 0

                # Restart environment (this also clears its explicit memory)
                next_state = env.reset()
                next_states[-1] = next_state

                # Check if there are enough total episode rewards to start checking if solved
                if len(reward_history) >= REWARD_WINDOW:

                    # Get mean reward of the last episode rewards
                    mean_reward = np.mean(reward_history[-REWARD_WINDOW:])

                    # Check if the environment is solved
                    if mean_reward >= STOP_REWARD and not solved:
                        print(f"Solved! Mean reward: {mean_reward:.3f}")
                        solved = True

        # Append results for each step to rollout tracking variables
        rollout_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=DEVICE))
        rollout_log_probs.append(log_probs)
        rollout_values.append(values.squeeze(-1))
        rollout_entropies.append(entropies)
        rollout_dones.append(torch.tensor(dones, dtype=torch.bool, device=DEVICE))

        # Set states to be next states for next iteration
        states = np.array(next_states,dtype=np.float32)

        global_step += NUM_ENVS

    state_t = torch.tensor(states,dtype=torch.float32,device=DEVICE)

    # Bootstrap value for the next state
    with torch.no_grad():
        _, next_values = net(state_t)

    next_values = next_values.squeeze(-1)

    # Initialize n-step return variables
    returns = []
    R = next_values

    # Go through rewards in rollout reversed to calculate discounted reward
    for step in reversed(range(REWARD_STEPS)):
        rewards = rollout_rewards[step]
        dones = rollout_dones[step]
        R = rewards + GAMMA * R * (~dones)
        returns.insert(0, R)

    # Stack tensors
    returns = torch.stack(returns)
    values = torch.stack(rollout_values)
    log_probs = torch.stack(rollout_log_probs)
    entropies = torch.stack(rollout_entropies)

    # Flatten dimensions
    returns = returns.reshape(-1)
    values = values.reshape(-1)
    log_probs = log_probs.reshape(-1)
    entropies = entropies.reshape(-1)

    # Calculate entropy beta
    progress = min(episode_count / NUM_EPISODES, 1.0)
    ENTROPY_BETA_CURRENT = (ENTROPY_BETA_START + progress * (ENTROPY_BETA_END - ENTROPY_BETA_START))

    # Calculate loss
    advantages = returns - values
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns.detach())
    entropy = entropies.mean()
    loss = (policy_loss + VALUE_COEF * value_loss - ENTROPY_BETA_CURRENT * entropy)

    # Take an optimizer step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),CLIP_GRADS)
    optimizer.step()

    # Write into tensorboard
    writer.add_scalar("loss/policy", policy_loss.item(), global_step)
    writer.add_scalar("loss/value", VALUE_COEF * value_loss.item(), global_step)
    writer.add_scalar("loss/entropy", ENTROPY_BETA_CURRENT * entropy.item(), global_step)
    writer.add_scalar("loss/total", loss.item(), global_step)

# Save the trained model
save_path = os.path.join(log_dir, "model.pt")
torch.save(net.state_dict(), save_path)
print("Model saved to", save_path)

writer.close()

print("Training finished.")