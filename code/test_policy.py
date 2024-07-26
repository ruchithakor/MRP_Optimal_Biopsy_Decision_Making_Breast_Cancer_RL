"""This python file testes both policies from backward induction and q learning
"""

# Import required packages
import numpy as np
import pandas as pd
from pathlib import Path

# Load the required data files
cwd_path = str(Path.cwd())
all_women_data = pd.read_csv(cwd_path + "/dataset/all_women_data.csv")
rewards_df = pd.read_csv(cwd_path + "/dataset/rewards.csv")
transitions_df = pd.read_csv(cwd_path + "/dataset/state_transition_probabilities.csv")

# Define the set of states, actions, rewards, and transition probabilities
S = np.sort(all_women_data["Risk Score"].unique())
A = ["AM", "B"]  # Actions: Annual Mammogram (AM) and Biopsy (B)
T = 60  # Time steps, e.g., from age 40 to 100

# Convert the rewards DataFrame into a dictionary
R = {row["age"]: {
        "Healthy": {"AM": row["Healthy_AM"], "B": row["Healthy_B"]},
        "Early": {"AM": row["Early_AM"], "B": row["Early_B"]},
        "Advanced": {"AM": row["Advanced_AM"], "B": row["Advanced_b"]}
    } for index, row in rewards_df.iterrows()}

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
    for state in S:  # Assuming states are numbered 1 through 100
        if state not in P[age]:
            P[age][state] = {"AM": {state: 1}, "B": {100: 1}}  # Self-loop for 'AM', transition to 100 for 'B'
        else:
            if state != 100:  # State 100 is absorbing
                remaining_prob = 1 - sum(P[age][state]["AM"].values())
                P[age][state]["AM"][state] = remaining_prob  # Self-loop probability

# Function to initialize the occupancy distribution
def initialize_occupancy_distribution(max_prob_state_1, max_prob_state_2):
    
    prob_state_1 = np.random.uniform(0, max_prob_state_1) # probability of early stage cancer
    prob_state_2 = np.random.uniform(0, max_prob_state_2) # probability of advanced stage cancer

    if prob_state_1 + prob_state_2 > 1:
        prob_state_2 = np.random.uniform(0, 1 - prob_state_1)
    
    prob_state_0 = 1.0 - prob_state_1 - prob_state_2 # probability of being healthy
    prob_state_0 = max(prob_state_0, 0) # make sure no negative probability
    
    return [prob_state_0, prob_state_1, prob_state_2, 0, 0] # return initial occupancy distribution


def perform_action(state, action, age):
    """function to perform the action
    """
    # get the next states and probabilities
    next_states = list(P[age][state][action].keys())
    probabilities = np.array(list(P[age][state][action].values()), dtype=float)

    probabilities = np.maximum(probabilities, 0) # make sure no negative probabilities
     
    probabilities /= probabilities.sum() # normalize probabilities to sum up to 1

    next_state = np.random.choice(next_states, p = probabilities) # choose next state
    reward = 0 # initialize the reward

    # asssign reward
    if next_state == 0:
        reward = float(R[age]["Healthy"][action])

    elif next_state in list(range(1, 21)):
        reward = float(R[age]["Early"][action])

    else:
        reward = float(R[age]["Advanced"][action])

    # return reward and next state
    return next_state, reward

# Function to test the policy
def test_policy(policy, num_episodes = 1000, max_prob_state_1 = 0.2, max_prob_state_2 = 0.3):
    
    results = []
    
    for _ in range(num_episodes):
        
        # initialize episodes for testing
        state = int((1 - initialize_occupancy_distribution(max_prob_state_1, max_prob_state_2)[0]) * 100) # get state

        total_reward = 0

        for age in range(40, 100):
            
            if state not in policy[age]:
                print(f"No policy for state {state} at time {age}. Defaulting to 'AM'.")
                action = "AM"
            else:
                action = policy[age][state]

            next_state, reward = perform_action(state, action, age) # perform action
            total_reward += reward # calculate total reward

            if next_state == 100:  # Absorbing state, exit the loop
                break

            state = next_state

        results.append(total_reward)

    # return mean rewards and std deviation
    return np.mean(results), np.std(results)

# Load the policy from backward induction data from the CSV file
bi_policy_df = pd.read_csv(cwd_path + "/results/backward_induction_policy.csv")

# Convert the policy DataFrame into a dictionary
policy_backward_induction = {}

for index, row in bi_policy_df.iterrows():

    t = row["time_stamp"]
    state = row["state"]
    optimal_action = row["action"]
    
    if t not in policy_backward_induction:
        policy_backward_induction[t] = {}

    policy_backward_induction[t][state] = optimal_action

ql_policy_df = pd.read_csv(cwd_path + "/results/qlearning_policy.csv")

# Convert the policy DataFrame into a dictionary
policy_q_learning = {}
for index, row in ql_policy_df.iterrows():

    t = row["time_stamp"]
    state = row["state"]
    optimal_action = row["action"]
    
    if t not in policy_q_learning:
        policy_q_learning[t] = {}

    policy_q_learning[t][state] = optimal_action

# Test the backward induction policy
mean_reward_bi, std_reward_bi = test_policy(policy_backward_induction)
print(f"Backward Induction Policy - Mean Reward: {round(mean_reward_bi, 4)}, Std Reward: {round(std_reward_bi, 4)}")

# Test the Q-learning policy
mean_reward_ql, std_reward_ql = test_policy(policy_q_learning)
print(f"Q-learning Policy - Mean Reward: {round(mean_reward_ql, 4)}, Std Reward: {round(std_reward_ql, 4)}")
