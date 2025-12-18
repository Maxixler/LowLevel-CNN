import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
import random
import time

# =============================================================================
# 1. OPTIMIZED MAZE SYSTEM
# =============================================================================

class PerfectMaze:
    def __init__(self, width=15, height=15, obstacle_density=0.1):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.start_pos = (0, 0)
        self.end_pos = (width - 1, height - 1)
        self._generate_perfect_maze(obstacle_density)
        
        # Precompute valid actions for O(1) access
        self.valid_actions = np.empty((height, width), dtype=object)
        self._precompute_actions()

    def _generate_perfect_maze(self, density):
        """Generates a guaranteed connected maze with optimized DFS."""
        self.grid.fill(0)
        
        # Add random obstacles
        n_obstacles = int(self.width * self.height * density)
        indices = np.random.choice(self.width * self.height, n_obstacles * 2, replace=False)
        
        count = 0
        for idx in indices:
            if count >= n_obstacles:
                break
            
            y, x = divmod(idx, self.width)
            
            # Don't block start or end
            if (x, y) == self.start_pos or (x, y) == self.end_pos:
                continue
                
            self.grid[y, x] = 1
            count += 1

        # Ensure Connectivity
        self._ensure_connectivity()

    def _ensure_connectivity(self):
        """Removes obstacles if they block the path to key areas."""
        while True:
            visited = np.zeros_like(self.grid, dtype=bool)
            queue = deque([self.start_pos])
            visited[self.start_pos[1], self.start_pos[0]] = True
            reachable_count = 0
            
            while queue:
                x, y = queue.popleft()
                reachable_count += 1
                
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if not visited[ny, nx] and self.grid[ny, nx] == 0:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
            
            # Check if all empty cells are reachable
            total_empty = np.sum(self.grid == 0)
            if reachable_count == total_empty:
                break
            
            # Identify unreachable cells
            unreachable_mask = (self.grid == 0) & (~visited)
            if not np.any(unreachable_mask):
                break
                
            # Pick a random unreachable cell
            uy, ux = np.where(unreachable_mask)
            if len(uy) > 0:
                # Remove a few random obstacles to open up paths
                obstacles = np.argwhere(self.grid == 1)
                if len(obstacles) > 0:
                    to_remove = obstacles[np.random.choice(len(obstacles), min(len(obstacles), 5), replace=False)]
                    for ry, rx in to_remove:
                        self.grid[ry, rx] = 0
            else:
                break

    def _precompute_actions(self):
        """Precomputes valid actions for every cell for O(1) lookup."""
        # Actions: 0:Right, 1:Down, 2:Left, 3:Up
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 1:
                    self.valid_actions[y, x] = np.array([], dtype=np.int8)
                    continue
                
                actions = []
                # Right
                if x + 1 < self.width and self.grid[y, x + 1] == 0:
                    actions.append(0)
                # Down
                if y + 1 < self.height and self.grid[y + 1, x] == 0:
                    actions.append(1)
                # Left
                if x - 1 >= 0 and self.grid[y, x - 1] == 0:
                    actions.append(2)
                # Up
                if y - 1 >= 0 and self.grid[y - 1, x] == 0:
                    actions.append(3)
                
                self.valid_actions[y, x] = np.array(actions, dtype=np.int8)

    def get_possible_actions(self, x, y):
        """O(1) action lookup."""
        return self.valid_actions[y, x]

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] == 0

# =============================================================================
# 2. OPTIMIZED SOLVER (Q-LEARNING)
# =============================================================================

class PerfectSolver:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0):
        self.maze = maze
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Q-Table: (y, x, action) - Optimized for fixed target (Exit)
        self.q_table = np.zeros((maze.height, maze.width, 4), dtype=np.float32)
        
        # Pre-allocate direction arrays for speed
        self.dx = np.array([1, 0, -1, 0], dtype=np.int8)
        self.dy = np.array([0, 1, 0, -1], dtype=np.int8)

    def get_action(self, x, y, tx, ty):
        """Epsilon-greedy action selection optimized."""
        possible_actions = self.maze.get_possible_actions(x, y)
        if len(possible_actions) == 0:
            return None
            
        if np.random.random() < self.epsilon:
            return np.random.choice(possible_actions)
        
        # Exploitation: Get max Q value actions
        q_values = self.q_table[y, x, possible_actions]
        max_q = np.max(q_values)
        
        best_indices = np.where(q_values == max_q)[0]
        chosen_idx = np.random.choice(best_indices)
        return possible_actions[chosen_idx]

    def update(self, x, y, action, reward, nx, ny, tx, ty):
        """Q-learning update rule."""
        current_q = self.q_table[y, x, action]
        
        # Max Q for next state
        next_actions = self.maze.get_possible_actions(nx, ny)
        if len(next_actions) > 0:
            max_next_q = np.max(self.q_table[ny, nx, next_actions])
        else:
            max_next_q = 0.0
            
        # Update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[y, x, action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# =============================================================================
# 3. HIGH-PERFORMANCE TRAINING LOOP
# =============================================================================

def train_agent(maze, agent, episodes=1000):
    print(f"Starting Training: {episodes} episodes")
    start_time = time.time()
    
    width, height = maze.width, maze.height
    dx_arr, dy_arr = agent.dx, agent.dy
    
    history_steps = []
    history_rewards = []
    
    for episode in range(episodes):
        # Select Start (Random) / Target (Fixed Exit)
        while True:
            sx, sy = np.random.randint(0, width), np.random.randint(0, height)
            tx, ty = width - 1, height - 1 # Fixed Target
            if maze.grid[sy, sx] == 0 and maze.grid[ty, tx] == 0 and (sx, sy) != (tx, ty):
                break
        
        cx, cy = sx, sy
        total_reward = 0
        steps = 0
        max_steps = width * height
        
        while steps < max_steps:
            action = agent.get_action(cx, cy, tx, ty)
            if action is None:
                break
                
            nx, ny = cx + dx_arr[action], cy + dy_arr[action]
            
            if nx == tx and ny == ty:
                reward = 100.0
                done = True
            else:
                reward = -1.0
                done = False
            
            agent.update(cx, cy, action, reward, nx, ny, tx, ty)
            
            cx, cy = nx, ny
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        agent.decay_epsilon()
        history_steps.append(steps)
        history_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_steps = np.mean(history_steps[-100:])
            print(f"Episode {episode+1:5d} | Avg Steps: {avg_steps:6.1f} | Epsilon: {agent.epsilon:.4f}")

    duration = time.time() - start_time
    print(f"Training Complete. Time: {duration:.2f}s, Speed: {episodes/duration:.1f} eps/s")
    return history_rewards

# =============================================================================
# 4. TESTING & VISUALIZATION
# =============================================================================

def test_agent(maze, agent, start, end):
    path = [start]
    cx, cy = start
    tx, ty = end
    visited = set([start])
    
    print(f"\nTesting Path: {start} -> {end}")
    
    steps = 0
    max_steps = maze.width * maze.height * 2
    
    while (cx, cy) != (tx, ty) and steps < max_steps:
        possible_actions = maze.get_possible_actions(cx, cy)
        if not len(possible_actions):
            print("Stuck!")
            break
            
        q_values = agent.q_table[cy, cx, possible_actions]
        best_action_idx = np.argmax(q_values)
        action = possible_actions[best_action_idx]
        
        nx, ny = cx + agent.dx[action], cy + agent.dy[action]
        
        if (nx, ny) in visited:
            print("Loop detected in test path!")
            pass
            
        cx, cy = nx, ny
        path.append((cx, cy))
        visited.add((cx, cy))
        steps += 1
    
    if (cx, cy) == (tx, ty):
        print(f"Target Reached in {steps} steps.")
        return path
    else:
        print("Failed to reach target.")
        return path

def visualize_path(maze, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(maze.grid, cmap='binary')
    
    if path:
        px, py = zip(*path)
        plt.plot(px, py, color='red', linewidth=3, marker='o', markersize=4, alpha=0.7)
        plt.plot(px[0], py[0], 'go', markersize=10, label='Start')
        plt.plot(px[-1], py[-1], 'bx', markersize=10, label='End')
    
    plt.title("Optimized Q-Learning Path")
    plt.legend()
    plt.grid(False)
    plt.savefig('result.png')
    print("Plot saved to result.png")

if __name__ == "__main__":
    # Configuration
    SIZE = 15
    DENSITY = 0.15
    EPISODES = 5000
    
    # Initialize
    maze = PerfectMaze(SIZE, SIZE, DENSITY)
    agent = PerfectSolver(maze, learning_rate=0.2, discount_factor=0.95)
    
    # Train
    train_agent(maze, agent, EPISODES)
    
    # Test
    start = (0, 0)
    end = (SIZE-1, SIZE-1)
    
    # Ensure start/end are valid
    if maze.grid[start[1], start[0]] == 1: maze.grid[start[1], start[0]] = 0
    if maze.grid[end[1], end[0]] == 1: maze.grid[end[1], end[0]] = 0
    
    path = test_agent(maze, agent, start, end)
    visualize_path(maze, path)