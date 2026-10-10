import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

class GridWorldEnv:
    # 0 = up, 1 = down, 2 = left, 3 = right
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, row_count=8, col_count=8, max_steps=50, wall_prob = 0.3, min_dist = 4, use_memory = True, visit_cap = 5):
        # Initialize parameters
        self.rows = row_count
        self.cols = col_count
        self.n_actions = 4
        self.wall_prob = wall_prob
        self.max_steps = max_steps
        self.min_dist = min_dist
        self.visit_cap = visit_cap

        # If False, the observation contains no explicit memory (plain MLP baseline)
        self.use_memory = use_memory

        # Set start and goal
        self.start = None
        self.goal = None

        # Start environment
        self.reset()

    def generate_random_walls(self, rows=8, cols=8, start=(0,0), goal=None, wall_prob=0.3):
        # Generate random walls for GridWorld while making sure a solution exists

        if goal is None:
            goal = (rows-1, cols-1)

        def path_exists(walls):
            # Check if there is a valid path using Breadth-First Search

            # BFS queue. Start searching from the agent's start position.
            queue = deque([start])

            # Keep track of positions we have already checked. This prevents repeatedly visiting the same position.
            visited = {start}

            # Continue searching while there are positions left to explore.
            while queue:

                # Take the next position from the front of the queue.
                r, c = queue.popleft()

                # If we reached the goal, then a valid path exists.
                if (r, c) == goal:
                    return True

                # All possible movements from the current position:
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

                # Check each neighboring position.
                for nr, nc in neighbors:
                    # A neighbor can be explored only if it is in the grid, not a wall and it has not been visited
                    if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls and (nr, nc) not in visited):
                        # Mark this position as visited.
                        visited.add((nr, nc))

                        # Add it to the queue so that its neighbors will also be explored.
                        queue.append((nr, nc))

            # If the queue becomes empty without reaching the goal, no valid path exists.
            return False

        # Keep generating random maps until we find one where the goal is reachable from the start.
        while True:

            # Set containing the coordinates of all walls.
            walls = set()

            # Go through every position in the grid.
            for r in range(rows):
                for c in range(cols):

                    # The start and goal positions must always remain free.
                    if (r, c) in {start, goal}:
                        continue

                    # Generate walls using given probability
                    if random.random() < wall_prob:
                        walls.add((r, c))

            # Check whether the randomly generated map is solvable.
            if path_exists(walls):

                # If it is solvable, return the walls.
                return walls

    def reset(self):
        # Set the start and goal
        while True:
            self.start = (random.randrange(self.rows), random.randrange(self.cols))
            self.goal = (random.randrange(self.rows), random.randrange(self.cols))
            dist = abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])
            if dist >= self.min_dist:
                break

        # Generate new walls
        self.walls = self.generate_random_walls(
            rows=self.rows, cols=self.cols,
            start=self.start, goal = self.goal,
            wall_prob=self.wall_prob)

        # Reset position and step counter
        self.position = self.start
        self.steps = 0

        # Reset explicit memory: visit counts per tile and the last action taken.
        # They are recreated here, so memory never leaks between episodes.
        self.visits = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.visits[self.start] = 1
        self.last_action = None

        # Get current state
        return self.get_state()

    def compute_action_mask(self):
        # Find all possible moves in the current state

        row, col = self.position
        mask = np.zeros(self.n_actions, dtype=np.bool_)

        # Check for every action if the target tile is inside the grid and not a wall
        for action, (d_row, d_col) in enumerate(self.MOVES):
            new_row = row + d_row
            new_col = col + d_col

            valid = (
                0 <= new_row < self.rows
                and 0 <= new_col < self.cols
                and (new_row, new_col) not in self.walls
            )

            mask[action] = valid

        return mask

    def get_action_mask(self):
        # Return the mask that was cached in get_state, so it is not computed twice per step
        return self.action_mask

    def get_state(self):
        # Get state observation containing only 4 surrounding tiles, current position and timer.

        # Get current position
        row, col = self.position

        # Get all possible actions and cache them for the training loop
        self.action_mask = self.compute_action_mask()

        # Get all surrounding walls
        surrounding_walls = (~self.action_mask).astype(np.float32)

        # Create base observation
        parts = [
            np.array([
                row / (self.rows - 1),
                col / (self.cols - 1),
                self.steps / self.max_steps
            ], dtype=np.float32),
            surrounding_walls
        ]

        # Add explicit memory features if enabled
        if self.use_memory:

            # Get how often each of the 4 neighbors was visited (0 for tiles outside the grid)
            neighbor_visits = np.zeros(4, dtype=np.float32)
            for action, (d_row, d_col) in enumerate(self.MOVES):
                new_row = row + d_row
                new_col = col + d_col
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    neighbor_visits[action] = min(self.visits[new_row, new_col], self.visit_cap) / self.visit_cap

            # Get how often the current tile was visited
            current_visits = np.array([min(self.visits[row, col], self.visit_cap) / self.visit_cap], dtype=np.float32)

            # Get the last action as one-hot vector (all zeros at the start of an episode)
            last_action = np.zeros(4, dtype=np.float32)
            if self.last_action is not None:
                last_action[self.last_action] = 1.0

            parts += [neighbor_visits, current_visits, last_action]

        return np.concatenate(parts)

    def step(self, action):
        # Get moving direction
        drow, dcol = self.MOVES[action]

        # Calculate the new position
        old_row, old_col = self.position
        new_row = old_row + drow
        new_col = old_col + dcol

        # Add steps and set reward
        self.steps += 1
        reward = -0.01

        # Check if current move is invalid
        invalid_move = (
            new_row < 0 or new_row >= self.rows or
            new_col < 0 or new_col >= self.cols or
            (new_row, new_col) in self.walls
        )

        # Penalize if invalid move and reset position to old position
        if invalid_move:
            new_row, new_col = old_row, old_col
            reward += -0.05

        # Set the new position
        self.position = (new_row, new_col)

        # Update explicit memory with the new position and the action taken
        self.visits[self.position] += 1
        self.last_action = action

        done = False
        finished = False

        # Check if current position is in the goal
        if self.position == self.goal:
            reward = 1
            done = True
            finished = True

        # Check if max steps are reached and if so terminate episode
        if self.steps >= self.max_steps and not done:
            done = True
            finished = False

        return self.get_state(), reward, done, finished


class GridWorldA2C(nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        hidden_size=176
    ):
        # hidden_size is chosen so the parameter count is close to the LSTM model
        # (LSTM with hidden size 64 has about 34k parameters). Check with count_parameters().
        super().__init__()

        input_size = int(input_shape[0])

        # Extract features from the current observation (no recurrent memory)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Actor and critic heads
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # Extract features
        features = self.network(x)

        # Calculate actor and critic predictions
        logits = self.actor(features)
        values = self.critic(features)

        return logits, values


def count_parameters(model):
    # Count all trainable parameters of a model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)