import tkinter as tk
import numpy as np
import time

class GridWorld:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.state = (0, 0)  
        self.goal_state = (grid_size-1, grid_size-1)
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = (0, 0)
        self.goal_state = (self.grid_size-1, self.grid_size-1)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size - 1:
            y += 1

        self.state = (x, y)
        reward = 1 if self.state == self.goal_state else -0.1
        done = self.state == self.goal_state

        self.move_goal()  # Move the goal after each step
        return self.state, reward, done

    def move_goal(self):
        gx, gy = self.goal_state
        move = np.random.choice(self.actions)
        if move == 'up' and gx > 0:
            gx -= 1
        elif move == 'down' and gx < self.grid_size - 1:
            gx += 1
        elif move == 'left' and gy > 0:
            gy -= 1
        elif move == 'right' and gy < self.grid_size - 1:
            gy += 1
        self.goal_state = (gx, gy)

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        x, y = self.state
        gx, gy = self.goal_state
        grid[x, y] = -1  
        grid[gx, gy] = 1  
        return grid

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            x, y = state
            return self.env.actions[np.argmax(self.q_table[x, y])]

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_idx = self.env.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        td_target = reward + self.gamma * self.q_table[next_x, next_y, best_next_action]
        td_error = td_target - self.q_table[x, y, action_idx]
        self.q_table[x, y, action_idx] += self.alpha * td_error

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class RLVisualizer(tk.Tk):
    def __init__(self, agent, env):
        super().__init__()
        self.agent = agent
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = 50
        self.title("RL Agent Visualizer")
        self.geometry(f"{self.grid_size * self.cell_size}x{self.grid_size * self.cell_size}")
        self.canvas = tk.Canvas(self, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size)
        self.canvas.pack()
        self.after(0, self.update_grid)

    def render_grid(self):
        self.canvas.delete("all")
        grid = self.env.render()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                if grid[i, j] == -1:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="blue")
                elif grid[i, j] == 1:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="green")
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

    def update_grid(self):
        state = self.env.reset()
        done = False
        while not done:
            self.render_grid()
            self.update_idletasks()
            self.update()
            time.sleep(1)  
            action = self.agent.choose_action(state)
            state, _, done = self.env.step(action)
        self.render_grid()

if __name__ == "__main__":
  env = GridWorld(grid_size=5)
  agent = QLearningAgent(env)
  agent.train(num_episodes=500)
    
  app = RLVisualizer(agent, env)
  app.mainloop()
