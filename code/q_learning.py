"""This python file implements the q learning algorithm to generate the policies and find biopsy threshold for each time-stamp(age).
"""

# Import required packages
import numpy as np
import pandas as pd
from pathlib import Path

# Define the set of states, actions, rewards, and transition probabilities
cwd_path = str(Path.cwd())
all_women_data = pd.read_csv(cwd_path + "/dataset/all_women_data.csv")
S = np.sort(all_women_data['Risk Score'].unique())

A = ['AM', 'B']  # Actions: Annual Mammogram (AM) and Biopsy (B)
T = 60  # Time steps, e.g., from age 40 to 100

# Read the rewards from the CSV file into a pandas DataFrame
rewards_df = pd.read_csv(cwd_path + "/dataset/rewards.csv")

# Convert the DataFrame into a dictionary with the required format for the MDP
# The dictionary will be structured as R[age][state][action]
R = {row["age"]: {
        "Healthy": {"AM": row["Healthy_AM"], "B": row["Healthy_B"]},
        "Early": {"AM": row["Early_AM"], "B": row["Early_B"]},
        "Advanced": {"AM": row["Advanced_AM"], "B": row["Advanced_b"]}
    } for index, row in rewards_df.iterrows()}

# Read the transition probabilities from the CSV file 'transitions.csv' is in the current working directory
transitions_df = pd.read_csv(cwd_path + "/dataset/state_transition_probabilities.csv")

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
    P[age][from_state]["B"][to_state] = 0 if to_state != 100 else 1 # If biopsy performed patient reaches state 100 and cycle closes

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
    """This function make sures that probabilitie are summed to 1 by normalizing
    """
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

# Initialize Q-table with zeros
Q = {age: {state: {action: 0 for action in A} for state in S} for age in range(40, 100)}

def update_Q(t_range, s_range, action):
    for t in t_range:
        for s in s_range:
            Q[t][s][action] = 1

# Update 'AM'
update_Q(range(40, 46), range(1, 5), "AM")
update_Q(range(46, 66), range(1, 9), "AM")
update_Q(range(66, 81), range(1, 11), "AM")
update_Q(range(81, 91), range(1, 16), "AM")
update_Q(range(91, 95), range(1, 18), "AM")
update_Q(range(95, 100), range(1, 21), "AM")

# Update 'B'
update_Q(range(40, 46), range(5, 101), "B")
update_Q(range(46, 66), range(9, 101), "B")
update_Q(range(66, 81), range(11, 101), "B")
update_Q(range(81, 91), range(16, 101), "B")
update_Q(range(91, 95), range(18, 101), "B")
update_Q(range(95, 100), range(21, 101), "B")

# set Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Exploration rate
epochs = 1000  # Number of epochs for training

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
    max_prob_state_1 = 0.2  # Example probability range for state 1
    max_prob_state_2 = 0.3  # Example probability range for state 2
    occupancy_distribution = initialize_occupancy_distribution(max_prob_state_1, max_prob_state_2)
    
    prob_0 = occupancy_distribution[0]
    initial_state_value = int((1 - prob_0) * 100)  # Scale to 0-100 range
    return initial_state_value

# Epsilon-greedy function
def epsilon_greedy(Q_values, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(A)
    else:
        action = max(Q_values, key=Q_values.get)
    return action

# Perform action function
def perform_action(state, action, age):

    if action == 'B':

        next_state = 100 # If action is biopsy, transition to state 100

        if state == 0: # reward for healthy 
            reward = R[age]["Healthy"][action]

        else:
            # Early and advance stages of cancer varies with age
            # assign rewards accordingly
            if age in list(range(40, 46)):
                if state in list(range(1, 5)):
                    reward = float(R[age]["Early"][action])
                else:
                    reward = float(R[age]["Advanced"][action]) 

            elif age in list(range(46, 66)):
                if state in list(range(1, 9)):
                    reward = float(R[age]["Early"][action]) 
                else:
                    reward = float(R[age]["Advanced"][action])

            elif age in list(range(66, 81)):
                if state in list(range(1, 11)):
                    reward = float(R[age]["Early"][action]) 
                else:
                    reward = float(R[age]["Advanced"][action])

            elif age in list(range(81, 91)):
                if state in list(range(1, 16)):
                    reward = float(R[age]["Early"][action]) 
                else:
                    reward = float(R[age]["Advanced"][action]) 

            elif age in list(range(91, 95)):
                if state in list(range(1, 18)):
                    reward = float(R[age]["Early"][action]) 
                else:
                    reward = float(R[age]["Advanced"][action]) 
            else:
                if state in list(range(1, 21)):
                    reward = float(R[age]["Early"][action]) 
                else:
                    reward = float(R[age]["Advanced"][action])
    
    else:

        # get the next states and probabilities
        next_states = list(P[age][state][action].keys())
        probabilities = list(P[age][state][action].values())
        
        # Handle cases with zero or missing probabilities
        if not probabilities or sum(probabilities) == 0:
            probabilities = [1 / len(next_states)] * len(next_states)  # Default to uniform distribution
        
        # Normalize probabilities
        probabilities = list(normalize_probabilities(dict(zip(next_states, probabilities))).values())
        
        # Validate probabilities
        probabilities = validate_probabilities(probabilities)
        
        next_state = np.random.choice(next_states, p = probabilities)
        if next_state == 100:
            reward = -100 # Terminal reward for cancer detection
        
        else:
            if state == 0:
                reward = R[age]["Healthy"][action]

            else:
                # Early and advance stages of cancer varies with age
                # assign rewards accordingly
                if age in list(range(40, 46)):
                    if state in list(range(1, 5)):
                        reward = float(R[age]["Early"][action])
                    else:
                        reward = float(R[age]["Advanced"][action]) 
                
                elif age in list(range(46, 66)):
                    if state in list(range(1, 9)):
                        reward = float(R[age]["Early"][action]) 
                    else:
                        reward = float(R[age]["Advanced"][action])
                
                elif age in list(range(66, 81)):
                    if state in list(range(1, 11)):
                        reward = float(R[age]["Early"][action]) 
                    else:
                        reward = float(R[age]["Advanced"][action])

                elif age in list(range(81, 91)):
                    if state in list(range(1, 16)):
                        reward = float(R[age]["Early"][action]) 
                    else:
                        reward = float(R[age]["Advanced"][action]) 

                elif age in list(range(91, 95)):
                    if state in list(range(1, 18)):
                        reward = float(R[age]["Early"][action]) 
                    else:
                        reward = float(R[age]["Advanced"][action])

                else:
                    if state in list(range(1, 21)):
                        reward = float(R[age]["Early"][action]) 
                    else:
                        reward = float(R[age]["Advanced"][action])
    
    return next_state, reward # return next state and reward

# Q-learning algorithm
for _ in range(epochs):
    for woman_id in range(1, 101):  # episodes for 100 women
        state = initial_state()  # Initial state for each woman
        
        for age in range(40, 100):

            action = epsilon_greedy(Q[age][state], epsilon) # choose action with greedy policy
            next_state, reward = perform_action(state, action, age) # get next state and reward

            next_max_q = max(Q[age + 1][next_state].values()) if (next_state != 100 and age != 99) else 0  # Terminal state has Q-value 0
            Q[age][state][action] += alpha * (reward + gamma * next_max_q - Q[age][state][action])
      
            state = next_state
    

# Extract the optimal policy from the Q-table
policy = {}
for age in range(40, 100):
    policy[age] = {}
    for state in S:
        policy[age][state] = max(Q[age][state], key = Q[age][state].get)

# Convert the Q-table to a DataFrame for easier inspection
q_table_data = []
for age in range(40, 100):
    for state in S:
        for action in A:
            q_table_data.append({'Time Step': age, 'State': state, 'Action': action, 'Q-Value': Q[age][state][action]})
q_table_df = pd.DataFrame(q_table_data)

# Save the Q-table data to a CSV file
q_table_filename = cwd_path + "/results/qlearning_table.csv"
q_table_df.to_csv(q_table_filename, index=False)

# Save the optimal policy to a CSV file
policy_data = []
for age in range(40, 100):
    for state in S:
        policy_data.append({'time_stamp': age, 'state': state, 'action': policy[age][state]})
q_policy_df = pd.DataFrame(policy_data)
policy_filename = cwd_path + "/results/qlearning_policy.csv"
q_policy_df.to_csv(policy_filename, index=False)


# Filter the DataFrame for rows where Action is 'B'
filtered_df = q_policy_df[q_policy_df['action'] == 'B']

# Group by 'Time Step' and get the first occurrence of 'B' for each time step
first_b_df = filtered_df.groupby('time_stamp').first().reset_index()

# Select the necessary columns
result_df = first_b_df[['time_stamp', 'state']]
q_threshold_filename = cwd_path + "/results/qlearning_threshold.csv"
result_df.to_csv(q_threshold_filename, index=False)
