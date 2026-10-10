import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

class GridWorldEnv:
    def __init__(self, row_count=8, col_count=8, max_steps=50, wall_prob = 0.3, min_dist = 4):
        # Initialize parameters
        self.rows = row_count
        self.cols = col_count
        self.n_actions = 4
        self.wall_prob = wall_prob
        self.max_steps = max_steps
        self.min_dist = min_dist

        # Set start and goal
        self.start = None
        self.goal = None

        # Define max steps and start environment
        self.max_steps = max_steps
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

        # Reset environment and get current state
        self.position = self.start
        self.steps = 0
        self.first_reward = True
        self.second_reward = True
        return self.get_state()


    def get_state(self):
        # Get state observation containing only 4 surrounding tiles and current position

        # Get current position
        row, col = self.position

        # Get all possible actions
        action_mask = self.get_action_mask()

        # Get all surrounding walls
        surrounding_walls = (~action_mask).astype(np.float32)

        # Create observation
        return np.concatenate([
            np.array([
                row / (self.rows - 1),
                col / (self.cols - 1),
                self.steps / self.max_steps
            ], dtype=np.float32),
            surrounding_walls
        ])


    def step(self, action):
        # 0 = up, 1 = down, 2 = left, 3 = right
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        # Get moving direction
        drow, dcol = moves[action]

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

        done = False
        finished = False

        # Check if current position is in the goal
        if self.position == self.goal:
            reward = 1
            self.lines_cleared = 1
            done = True
            finished = True

        # Check if max steps are reached and if so terminate episode
        if self.steps >= self.max_steps and not done:
            done = True
            finished = False

        return self.get_state(), reward, done, finished

    def get_action_mask(self):
        # Find all possible moves in the current state
        
        moves = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        row, col = self.position
        mask = np.zeros(self.n_actions, dtype=np.bool_)

        for action, (d_row, d_col) in enumerate(moves):
            new_row = row + d_row
            new_col = col + d_col

            valid = (
                0 <= new_row < self.rows
                and 0 <= new_col < self.cols
                and (new_row, new_col) not in self.walls
            )

            mask[action] = valid

        return mask


class GridWorldA2C(nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        hidden_size=64
    ):
        super().__init__()

        input_size = int(input_shape[0])

        # Extract features from the current observation
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh()
        )

        # Recurrent memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Actor and critic heads
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self.hidden_size = hidden_size

    def initial_hidden(self, batch_size, device):
        # Initialize the LSTM memory vectors
        
        h = torch.zeros(1, batch_size, 
                        self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, 
                        self.hidden_size, device=device)
        return (h, c)

    def forward(self, x, hidden):

        # Extract features
        features = self.network(x)

        # Add sequence dimension: [batch, 1, hidden_size]
        features = features.unsqueeze(1)

        # LSTM forward pass
        output, hidden = self.lstm(features, hidden)

        # Remove sequence dimension
        output = output.squeeze(1)

        # Calculate actor and critic predictions
        logits = self.actor(output)
        values = self.critic(output)

        return logits, values, hidden

