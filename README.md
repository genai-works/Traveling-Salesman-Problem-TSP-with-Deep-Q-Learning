# **Traveling Salesman Problem (TSP) with Deep Q-Learning**

## **Project Overview**

The **Traveling Salesman Problem (TSP)** is a classic optimization problem where a salesman must visit a given set of cities exactly once and return to the starting city. The goal is to find the shortest possible route that covers all cities. TSP is an NP-hard problem, meaning the time it takes to solve grows exponentially with the number of cities.

In this project, we tackle TSP using **Reinforcement Learning (RL)**, specifically leveraging a **Deep Q-Network (DQN)** to learn an optimal path between cities. The Q-network helps the agent (salesman) learn which city to visit next based on the distances between cities, minimizing the total travel distance.

## **Project Structure**

This project is implemented using **Python** and **PyTorch** for building the neural network model. The code is broken down into several sections to handle the following tasks:

1. **City Class**: Represents each city by its x and y coordinates and calculates distances between cities.
2. **Deep Q-Network (DQN)**: A neural network that approximates Q-values for each action (next city to visit) based on the current state (visited cities).
3. **TSP Agent**: An RL agent that interacts with the environment to learn the shortest path by choosing which city to visit next. The agent stores experiences and trains using Q-learning.
4. **TSP Environment**: Simulates the TSP environment where the agent can take actions (visit cities) and receive rewards based on travel distance. The goal is to minimize negative rewards (distance).
5. **Training Loop**: The agent is trained over multiple episodes, gradually improving its ability to find shorter paths between cities.

---

## **How to Use This Project**

### **Requirements**

1. **Python 3.7+**
2. **PyTorch** (Deep learning framework)
3. **Matplotlib** (For plotting the results)

You can install the required libraries using:

```bash
pip install torch numpy matplotlib
```

### **Running the Code**

1. **Define Cities**: The cities are represented by their x and y coordinates. The code randomly generates cities with coordinates in a 2D plane. 

2. **Train the Agent**: The agent is trained using **Deep Q-Learning** over several episodes. The goal is for the agent to learn the optimal route that minimizes the total travel distance between cities. You can run the training process by calling the function `train_tsp_agent()`.

3. **Plot the Solution**: After training, the agent will have learned a path. You can visualize this path using `plot_solution()`, which plots the cities and the route learned by the agent.

### **Example**

To run the program, you can define the cities and train the agent as follows:

```python
# Example execution: Define a set of random cities and train the TSP agent

# Generate random cities
num_cities = 10
cities = [City(x=random.randint(0, 100), y=random.randint(0, 100)) for _ in range(num_cities)]

# Train the Q-learning agent for 500 episodes
train_tsp_agent(cities, episodes=500)

# Plot the learned solution
plot_solution(cities, learned_path)  # Plot the final path after training
```

### **Important Functions**

- **`City(x, y)`**: Creates a city with x and y coordinates.
- **`TSPAgent`**: Implements the agent, including decision-making and learning from experience.
- **`TSPEnvironment`**: Simulates the environment where the agent takes actions (visits cities).
- **`train_tsp_agent(cities, episodes)`**: Trains the agent on the defined set of cities for a specified number of episodes.
- **`plot_solution(cities, path)`**: Plots the route learned by the agent after training.

---

## **Project Details**

### **Deep Q-Learning**

The project implements **Deep Q-Learning (DQN)**, which is a reinforcement learning method. The agent (salesman) learns by interacting with the environment (TSP problem):

1. **State**: Represents which cities have been visited.
2. **Action**: The next city to visit.
3. **Reward**: Negative distance between cities. The closer the cities, the better the reward (since the agent aims to minimize total travel distance).
4. **Q-Network**: A neural network that approximates the Q-values for each action in a given state. The agent selects actions (cities to visit) based on these Q-values.
5. **Epsilon-Greedy Strategy**: The agent balances exploration (random city selection) and exploitation (selecting the best known action based on Q-values).

### **Key Concepts**

- **Experience Replay**: The agent stores past experiences (state transitions) in memory and replays them for learning. This helps stabilize learning by breaking correlation between consecutive actions.
- **Discount Factor (γ)**: Used to balance the importance of future rewards. A higher value makes the agent more farsighted, considering future rewards more heavily.
- **Epsilon Decay**: Over time, the agent reduces the rate at which it explores random actions. This helps the agent exploit its learned knowledge more often in later episodes.

---

## **Customization**

You can modify the following aspects of the project:
1. **Number of Cities**: Change the number of cities by adjusting the `num_cities` variable.
2. **Hyperparameters**: Adjust parameters like `gamma`, `epsilon`, `epsilon_decay`, and `learning_rate` in the `TSPAgent` class to tweak the learning behavior.
3. **Episodes**: The number of training episodes can be increased or decreased depending on the complexity of the problem.

---

## **Results and Visualization**

After training, you can visualize the path the agent has learned using the `plot_solution()` function. It shows the route that visits all cities exactly once and returns to the starting city.

---

## **Future Improvements**

1. **Larger Problem Instances**: For larger instances of TSP (many cities), more advanced techniques or increased training episodes may be required.
2. **Hybrid Approaches**: Combining genetic algorithms or other optimization techniques with Q-learning might lead to better results.
3. **Parallelization**: For larger city sets, parallelizing the experience replay and training can speed up learning.

---

## **Conclusion**

This project demonstrates how **Reinforcement Learning** can be used to tackle classic optimization problems like the **Traveling Salesman Problem**. The Q-learning agent gradually learns to minimize the total travel distance by exploring different routes and improving its policy over time.

By adjusting the parameters and experimenting with different setups, you can explore how the agent's performance evolves and how quickly it converges to an optimal solution.

