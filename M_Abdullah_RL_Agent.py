import tkinter as tk
import numpy as np
import time
from PIL import Image, ImageTk  # Import PIL for image handling


class GridWorldMDP:
    def __init__(self, grid_size=7, obstacle_prob=0.2):
        self.grid_size = grid_size
        self.actions = ['up', 'down', 'left', 'right']  
        self.state_space = [(i, j) for i in range(grid_size) for j in range(grid_size)] 
        self.obstacles = self._generate_obstacles(obstacle_prob)  # Generating obstacles
        self.reset()

    def _generate_obstacles(self, prob):
        obstacles = set()
        for state in self.state_space:
            if np.random.random() < prob and state not in [(0, 0), (self.grid_size-1, self.grid_size-1)]:       #generating obstacles within our grid environment, and having probability of generating, max 1
                obstacles.add(state)        #pushed into obstacles set
        return obstacles


    def reset(self):
        self.state = (0, 0)  # Start state
        self.goal_state = (self.grid_size-1, self.grid_size-1) 
        return self.state

    
    def step(self, action):
        if np.random.random() < 0.8:  # 80% chance of taking the intended action
            next_state = self._get_next_state(self.state, action)
        else:  # 20% chance of taking a random action
            next_state = self._get_next_state(self.state, np.random.choice(self.actions))

        # Check if the next state is an obstacle
        if next_state in self.obstacles:
            next_state = self.state  # Stay in the current state

        self.state = next_state
        self._move_goal()  # Randomly move the goal
        
        reward = self._get_reward()  # Get the reward for the current state
        done = (self.state == self.goal_state)  

        return self.state, reward, done

 
    def _get_next_state(self, state, action):
        x, y = state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size - 1:
            y += 1
        return (x, y)

  
    #change reward function too based on how far(distance), goal state is to teach our agent 
    def _get_reward(self):
        if self.state == self.goal_state:
            return 10
        elif self.state in self.obstacles:
            return -5
        else:
            return -0.1 - 0.1 * (abs(self.state[0] - self.goal_state[0]) + abs(self.state[1] - self.goal_state[1]))   

    
    def _move_goal(self):
        gx, gy = self.goal_state
        move = np.random.choice(self.actions)           #random choice for movement of goal
        new_goal = self._get_next_state((gx, gy), move)     
        if new_goal not in self.obstacles:
            self.goal_state = new_goal

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for obstacle in self.obstacles:
            grid[obstacle] = -1
        x, y = self.state
        gx, gy = self.goal_state
        grid[x, y] = 1
        grid[gx, gy] = 2
        return grid


class QLearningAgent:
    def __init__(self, mdp, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.mdp = mdp
        self.alpha = alpha  
        self.gamma = gamma 
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.q_table = {}  # Q-table to store state-action values
        for state in mdp.state_space:               #Implemented Markov Decision Process based learning agent
            for goal in mdp.state_space:
                self.q_table[(state, goal)] = {a: 0 for a in mdp.actions}       #initially q_table is set to 0 for all states 

    def choose_action(self, state, goal_state):
        # Decide whether to explore or exploit
    
        if np.random.rand() < self.epsilon:                         #Started with full exploration and then with time balancing 
            # Explore: choose a random action                       exploration and exploitation, so as we can achieve exploitation 
            return np.random.choice(self.mdp.actions)               #at maxsimum
        else:
            # Exploit: choose the best action based on Q-values
            best_action = self.mdp.actions[0]
            best_value = self.q_table[(state, goal_state)][best_action]
            
            for action in self.mdp.actions[1:]:
                action_value = self.q_table[(state, goal_state)][action]
                if action_value > best_value:
                    best_action = action
                    best_value = action_value
            
            return best_action

    def update_q_table(self, state, action, reward, next_state, goal_state):
        # Get the current Q-value
        current_q = self.q_table[(state, goal_state)][action]
        
        #findind the maximum Q-value for the next state.
        #iterating through all possible actions from the next state
        # and selects the highest Q-value
        #here we're using epsilon greedy approach to update the Q-value
        #based on the best possible future action of our agent. 

        max_next_q = self.q_table[(next_state, goal_state)][self.mdp.actions[0]]
        for next_action in self.mdp.actions[1:]:
            next_q = self.q_table[(next_state, goal_state)][next_action]
            if next_q > max_next_q:
                max_next_q = next_q
        
        #same formula used for calculations of new Q-Value
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        
        #Updating
        self.q_table[(state, goal_state)][action] = new_q

    # Training agent by interacting with the environment for a given number of episodes i.e., 5000 for now. 
    def train(self, num_episodes=5000):
        for episode in range(num_episodes):
            state = self.mdp.reset()
            done = False
            while not done:
                action = self.choose_action(state, self.mdp.goal_state)
                next_state, reward, done = self.mdp.step(action)
                self.update_q_table(state, action, reward, next_state, self.mdp.goal_state)
                state = next_state
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Decay exploration rate, and struggling to maxsimize exploitation at any given state 
            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {self.epsilon:.2f}")


class RLVisualizer(tk.Tk):
    def __init__(self, agent, mdp):
        super().__init__()
        self.agent = agent
        self.mdp = mdp
        self.grid_size = mdp.grid_size
        self.cell_size = 50 
        self.title("Tom & Jerry")
        self.geometry(f"{self.grid_size * self.cell_size}x{self.grid_size * self.cell_size}")
        self.canvas = tk.Canvas(self, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size)
        self.canvas.pack()
        
        # Load and resize the sprite images
        self.player_sprite = Image.open("catpic.png")
        self.player_sprite = self.player_sprite.resize((self.cell_size, self.cell_size), Image.LANCZOS)     #resamping my sprites so that it converge in grid cells 
                                                                                                            #Lanczos is filter in PIL lib
        self.player_sprite = ImageTk.PhotoImage(self.player_sprite)

        self.goal_sprite = Image.open("mousepic.png")
        self.goal_sprite = self.goal_sprite.resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.goal_sprite = ImageTk.PhotoImage(self.goal_sprite)

        self.obstacle_sprite = Image.open("dead_state.png")
        self.obstacle_sprite = self.obstacle_sprite.resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.obstacle_sprite = ImageTk.PhotoImage(self.obstacle_sprite)
        
        self.after(0, self.update_grid)

    # Render the grid on the canvas
    def render_grid(self):
        self.canvas.delete("all")
        grid = self.mdp.render()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                if grid[i, j] == -1:
                    self.canvas.create_image(x0 + self.cell_size//2, y0 + self.cell_size//2, image=self.obstacle_sprite)  # Obstacle sprite
                elif grid[i, j] == 1:
                    self.canvas.create_image(x0 + self.cell_size//2, y0 + self.cell_size//2, image=self.player_sprite)  # Player sprite
                elif grid[i, j] == 2:
                    self.canvas.create_image(x0 + self.cell_size//2, y0 + self.cell_size//2, image=self.goal_sprite)  # Goal sprite
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

    # Update the grid periodically to visualize the agent's movement
    def update_grid(self):
        state = self.mdp.reset()
        steps = 0
        max_steps = 100
        while steps < max_steps:
            self.render_grid()
            self.update_idletasks()
            self.update()
            time.sleep(0.2)
            action = self.agent.choose_action(state, self.mdp.goal_state)
            state, reward, done = self.mdp.step(action)
            steps += 1
            if done:
                print(f"Goal reached in {steps} steps!")
                break
        if steps == max_steps:
            print("Max steps reached without catching the goal.")
        self.render_grid()
        self.after(1000, self.update_grid)  # Restart after 1 second

# Main function to run the training and visualization
if __name__ == "__main__":
    mdp = GridWorldMDP(grid_size=7, obstacle_prob=0.2)
    q_agent = QLearningAgent(mdp)
    print("Training the agent...")
    q_agent.train(num_episodes=5000)
    print("Training complete. Starting visualization...")
    app = RLVisualizer(q_agent, mdp)
    app.mainloop()
