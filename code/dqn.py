"""This python file implements the DQN algorithm to generate the policies and find biopsy threshold for each time-stamp(age).
"""

# Import required packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import random
from tqdm import tqdm
from pathlib import Path

# Define the set of states, actions, rewards, and transition probabilities
cwd_path = str(Path.cwd())
all_women_data = pd.read_csv(cwd_path + "/dataset/all_women_data_v2.csv")
S = np.sort(all_women_data['Risk Score'].unique())

A = ["AM", "B"] # Actions: Annual Mammogram (AM) and Biopsy (B)
T = 60  # Time steps, e.g., from age 40 to 100

# Read the rewards from the CSV file into a pandas DataFrame
rewards_df = pd.read_csv(cwd_path + "/dataset/rewards.csv")

# Convert the DataFrame into a dictionary with the required format for the MDP
# The dictionary will be structured as R[age][state][action]
R = {row["age"]: {
        "Healthy": {"AM": row["Healthy_AM"], "B": row["Healthy_B"]},
        "Early": {"AM": row["Early_AM"], "B": row["Early_B"]},
        "Advanced": {"AM": row["Advanced_AM"], "B": row["Advanced_b"]}
    } for _, row in rewards_df.iterrows()}

# Read the transition probabilities from the CSV file 'transitions.csv' is in the current working directory
transitions_df = pd.read_csv(cwd_path + "/dataset/state_transition_probabilities_v2.csv")

# Initialize the transition probabilities dictionary for each age
P = {age: {} for age in range(40, 100)}

# Populate the dictionary with data from the DataFrame
for ind, row in transitions_df.iterrows():

    age = int(row["Age"])
    from_state = int(row["From State"])
    to_state = int(row["To State"])
    probability = row["Probability"]
    
    if from_state not in P[age]:
        P[age][from_state] = {"AM": {}, "B": {}}  # Assuming 'B' always leads to state 100
    
    # Assign the probability for transitioning from 'from_state' to 'to_state' under action 'AM'
    P[age][from_state]["AM"][to_state] = probability
    P[age][from_state]["B"][to_state] = 0 if to_state != 100 else 1

# Ensure all states have transition probabilities for all ages
for age in range(40, 100):
    for state in S: 
        if state not in P[age]:
            P[age][state] = {"AM": {state: 1}, "B": {100: 1}}  # Self-loop for 'AM', transition to 100 for 'B'
        else:
            if state != 100:  # State 100 is absorbing
                remaining_prob = 1 - sum(P[age][state]["AM"].values())
                P[age][state]["AM"][state] = remaining_prob  # Self-loop probability

# Function to check and normalize transition probabilities
def normalize_probabilities(transition_probs):
    transition_probs = {state: max(prob, 0) for state, prob in transition_probs.items()}  # make sure there are no negative probabilities
    total_prob = sum(transition_probs.values())
    if total_prob == 0:
        return transition_probs  # Avoid division by zero
    return {state: prob / total_prob for state, prob in transition_probs.items()}

# Normalize the transition probabilities for each state-action pair
for age in P:
    for state in P[age]:
        for action in P[age][state]:
            P[age][state][action] = normalize_probabilities(P[age][state][action])

# Function to validate transition probabilities
def validate_probabilities(probabilities):
    total_prob = sum(probabilities)
    if not np.isclose(total_prob, 1.0):
        raise ValueError(f"Probabilities do not sum to 1: {probabilities} (sum = {total_prob})")
    if any(prob < 0 for prob in probabilities):
        raise ValueError(f"Probabilities are not non-negative: {probabilities}")
    return probabilities

action_map = {
    'AM': 0,
    'B': 1
}

reverse_action_map = {v: k for k, v in action_map.items()}

# Hyperparameters
alpha = 0.001  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Exploration rate
epsilon_min = 0.01  # Minimum epsilon
epsilon_decay = 0.001  # Decay rate for epsilon
epochs = 1000  # Number of epochs for training
batch_size = 64  # Batch size for experience replay
memory_size = 10000  # Experience replay buffer size
target_update_frequency = 10  # Frequency to update target network

# Neural Network for the Q-Function
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Initialize DQNs
state_size = 1  # Single state input (Risk Score)
action_size = len(A)  # Two actions (AM, B)
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
memory = deque(maxlen=memory_size)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(A)
    state = torch.tensor([state], dtype=torch.float32)
    with torch.no_grad():
        action_values = policy_net(state)
    return np.argmax(action_values.numpy())

def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        state = torch.tensor([state], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        
        # Check if the action is already an integer
        if isinstance(action, str):
            action_idx = action_map[action]  # Convert action string to index
        else:
            action_idx = action  # Action is already an integer

        # Predict Q-values for the current state using the policy network
        q_values = policy_net(state)
        current_q = q_values.squeeze()[action_idx]  # Get the Q-value for the chosen action

        if done:
            target_q = reward
        else:
            # Predict Q-values for next state using the target network
            with torch.no_grad():
                next_q_values = target_net(next_state)
                target_q = reward + gamma * torch.max(next_q_values)

        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def update_target_net():
    target_net.load_state_dict(policy_net.state_dict())

# Initialize occupancy distribution function for episodes
def initialize_occupancy_distribution(max_prob_state_1, max_prob_state_2):
    prob_state_1 = np.random.uniform(0, max_prob_state_1) # probability of early stage cancer
    prob_state_2 = np.random.uniform(0, max_prob_state_2) # probability of advanced stage cancer
    
    if prob_state_1 + prob_state_2 > 1:
        prob_state_2 = np.random.uniform(0, 1 - prob_state_1)

    prob_state_0 = 1.0 - prob_state_1 - prob_state_2 # probability of being healthy
    prob_state_0 = max(prob_state_0, 0) # make sure probability is not negative

    return [prob_state_0, prob_state_1, prob_state_2, 0, 0] # return occupancy matrics

def initial_state():
    max_prob_state_1 = 0.3
    max_prob_state_2 = 0.4
    occupancy_distribution = initialize_occupancy_distribution(max_prob_state_1, max_prob_state_2)
    prob_0 = occupancy_distribution[0]
    initial_state_value = int((1 - prob_0) * 100)
    return initial_state_value

def perform_action(state, action, age):
    done = False
    if action == 1:  # 'B' corresponds to action index 1
        next_state = 100
        done = True

        if state == 0:
            reward = R[age]["Healthy"]['B']
        else:
            if age in range(40, 46):
                reward = R[age]["Early"]['B'] if state in range(1, 5) else R[age]["Advanced"]['B']
            elif age in range(46, 66):
                reward = R[age]["Early"]['B'] if state in range(1, 9) else R[age]["Advanced"]['B']
            elif age in range(66, 81):
                reward = R[age]["Early"]['B'] if state in range(1, 11) else R[age]["Advanced"]['B']
            elif age in range(81, 91):
                reward = R[age]["Early"]['B'] if state in range(1, 16) else R[age]["Advanced"]['B']
            elif age in range(91, 95):
                reward = R[age]["Early"]['B'] if state in range(1, 18) else R[age]["Advanced"]['B']
            else:
                reward = R[age]["Early"]['B'] if state in range(1, 21) else R[age]["Advanced"]['B']
    else:
        next_states = list(P[age][state]['AM'].keys())
        probabilities = list(P[age][state]['AM'].values())
        if not probabilities or sum(probabilities) == 0:
            probabilities = [1 / len(next_states)] * len(next_states)
        probabilities = list(normalize_probabilities(dict(zip(next_states, probabilities))).values())
        probabilities = validate_probabilities(probabilities)
        next_state = np.random.choice(next_states, p=probabilities)
        reward = -100 if next_state == 100 else R[age]["Healthy"]['AM']

        if state != 0:
            if age in range(40, 46):
                reward = R[age]["Early"]['AM'] if state in range(1, 5) else R[age]["Advanced"]['AM']
            elif age in range(46, 66):
                reward = R[age]["Early"]['AM'] if state in range(1, 9) else R[age]["Advanced"]['AM']
            elif age in range(66, 81):
                reward = R[age]["Early"]['AM'] if state in range(1, 11) else R[age]["Advanced"]['AM']
            elif age in range(81, 91):
                reward = R[age]["Early"]['AM'] if state in range(1, 16) else R[age]["Advanced"]['AM']
            elif age in range(91, 95):
                reward = R[age]["Early"]['AM'] if state in range(1, 18) else R[age]["Advanced"]['AM']
            else:
                reward = R[age]["Early"]['AM'] if state in range(1, 21) else R[age]["Advanced"]['AM']

    return next_state, reward, done

# Training Loop
for epoch in tqdm(range(epochs)):
    for woman_id in range(1, 101):
        state = initial_state()
        for age in range(40, 100):
            action = act(state, epsilon)
            next_state, reward, done = perform_action(state, action, age)
            remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        replay()
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    if epoch % target_update_frequency == 0:
        update_target_net()

# Extract the optimal policy from the Q-table
policy = {}
for age in range(40, 100):
    policy[age] = {}
    for state in S:
        state_tensor = torch.tensor([state], dtype=torch.float32)
        action = np.argmax(policy_net(state_tensor).detach().numpy())
        policy[age][state] = A[action]

# Save the optimal policy to a CSV file
policy_data = []
for age in range(40, 100):
    for state in S:
        policy_data.append({'time_stamp': age, 'state': state, 'action': policy[age][state]})
q_policy_df = pd.DataFrame(policy_data)
policy_filename = cwd_path + "/results/dqn_qlearning_policy.csv"
q_policy_df.to_csv(policy_filename, index=False)