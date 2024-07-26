import numpy as np
import pandas as pd

# Load the data
all_women_data = pd.read_csv("/Users/ruchithakor/Downloads/Masters_Docs/MRP/Implementation/v3/all_women_data_v2.csv")
rewards_df = pd.read_csv("/Users/ruchithakor/Downloads/Masters_Docs/MRP/Implementation/v3/rewards.csv")
transitions_df = pd.read_csv('/Users/ruchithakor/Downloads/Masters_Docs/MRP/Implementation/v3/state_transition_probabilities_v2.csv')

# Define the set of states, actions, rewards, and transition probabilities
# S = np.sort(all_women_data['Risk Score'].unique())
S = np.round(np.arange(0, 100.1, 0.1), 1)
A = ['AM', 'B']  # Actions: Annual Mammogram (AM) and Biopsy (B)
T = 60  # Time steps, e.g., from age 40 to 99

# Convert the rewards DataFrame into a dictionary
R = {row['t']: {
        'Healthy': {'AM': row['Healthy_AM'], 'B': row['Healthy_B']},
        'Early': {'AM': row['Early_AM'], 'B': row['Early_B']},
        'Advanced': {'AM': row['Advanced_AM'], 'B': row['Advanced_b']}
    } for index, row in rewards_df.iterrows()}

# Initialize the transition probabilities dictionary for each age
P = {t: {} for t in range(40, 100)}

# Populate the dictionary with data from the DataFrame
for index, row in transitions_df.iterrows():
    age = int(row['Age'])
    from_state = int(row['From State'])
    to_state = int(row['To State'])
    probability = row['Probability']
    
    if from_state not in P[age]:
        P[age][from_state] = {'AM': {}, 'B': {}}  # Assuming 'B' always leads to state 100
    
    # Assign the probability for transitioning from 'from_state' to 'to_state' under action 'AM'
    P[age][from_state]['AM'][to_state] = probability
    P[age][from_state]['B'][to_state] = probability

# Ensure all states have transition probabilities for all ages
for t in range(40, 100):
    for s in S:  # Assuming states are numbered 1 through 100
        if s not in P[t]:
            P[t][s] = {'AM': {s: 1}, 'B': {100: 1}}  # Self-loop for 'AM', transition to 100 for 'B'
        else:
            if s != 100:  # State 100 is absorbing
                remaining_prob = 1 - sum(P[t][s]['AM'].values())
                P[t][s]['AM'][s] = remaining_prob  # Self-loop probability

# Function to initialize the occupancy distribution
def initialize_occupancy_distribution(max_prob_state_1, max_prob_state_2):
    prob_state_1 = np.random.uniform(0, max_prob_state_1)
    prob_state_2 = np.random.uniform(0, max_prob_state_2)
    if prob_state_1 + prob_state_2 > 1:
        prob_state_2 = np.random.uniform(0, 1 - prob_state_1)
    prob_state_0 = 1.0 - prob_state_1 - prob_state_2
    prob_state_0 = max(prob_state_0, 0)
    return [prob_state_0, prob_state_1, prob_state_2, 0, 0]


def perform_action(state, action, age):
    next_states = list(P[age][state][action].keys())
    # probabilities = np.round(list(P[age][state][action].values()), 2)
    probabilities = np.array(list(P[age][state][action].values()), dtype=float)
    # print(probabilities)
    probabilities = np.maximum(probabilities, 0)

    # Check for NaN values
    nan_mask = np.isnan(probabilities)

    # Replace NaN values with 0
    probabilities[nan_mask] = 0
    # print(probabilities)
    probabilities /= probabilities.sum()

    next_state = np.random.choice(next_states, p=probabilities)
    reward = 0
    if next_state == 0:
        reward = float(R[age]['Healthy'][action])
    elif next_state in list(range(1, 21)):
        reward = float(R[age]['Early'][action])
    else:
        reward = float(R[age]['Advanced'][action])
    return next_state, reward


# Function to test the policy
def test_policy(policy, num_episodes=1000, max_prob_state_1=0.2, max_prob_state_2=0.2):
    results = []
    for episode in range(num_episodes):
        state = round((1 - initialize_occupancy_distribution(max_prob_state_1, max_prob_state_2)[0]) * 100, 1)
        total_reward = 0
        for t in range(40, 100):
            if state not in policy[t]:
                print(f"No policy for state {state} at time {t}. Defaulting to 'AM'.")
                action = 'AM'
            else:
                action = policy[t][state]
            next_state, reward = perform_action(state, action, t)
            total_reward += reward
            if next_state == 100:  # Absorbing state, exit the loop
                break
            state = next_state
        results.append(total_reward)
    return np.mean(results), np.std(results)

# Assuming `policy_backward_induction` and `policy_q_learning` are the policies obtained from backward induction and Q-learning respectively
# Load the policy data from the CSV file
bi_policy_df = pd.read_csv("/Users/ruchithakor/Downloads/Masters_Docs/MRP/Implementation/v3/results_data.csv")

# Convert the policy DataFrame into a dictionary
policy_backward_induction = {}
for index, row in bi_policy_df.iterrows():
    t = row['time_stamp']
    state = row['state']
    optimal_action = row['optimal_action']
    
    if t not in policy_backward_induction:
        policy_backward_induction[t] = {}
    policy_backward_induction[t][state] = optimal_action

ql_policy_df = pd.read_csv("/Users/ruchithakor/Downloads/Masters_Docs/MRP/Implementation/v3/policy.csv")

# Convert the policy DataFrame into a dictionary
policy_q_learning = {}
for index, row in ql_policy_df.iterrows():
    t = row['Time Step']
    state = row['State']
    optimal_action = row['Action']
    
    if t not in policy_q_learning:
        policy_q_learning[t] = {}
    policy_q_learning[t][state] = optimal_action

# Test the backward induction policy
mean_reward_bi, std_reward_bi = test_policy(policy_backward_induction)
print(f"Backward Induction Policy - Mean Reward: {mean_reward_bi}, Std Reward: {std_reward_bi}")

# Test the Q-learning policy
mean_reward_ql, std_reward_ql = test_policy(policy_q_learning)
print(f"Q-learning Policy - Mean Reward: {mean_reward_ql}, Std Reward: {std_reward_ql}")
