"""This python file implements the Backward Induction algorithm to generate the policies and find biopsy threshold for each time-stamp(age)
at granular level of states.
"""

# import required packages
import numpy as np
import pandas as pd

# Define the set of states, actions, rewards, and transition probabilities
transitions_df = pd.read_csv("/Users/ruchithakor/Downloads/Masters_Docs/MRP/MRP_Optimal_Biopsy_Decision_Making_Breast_Cancer_RL/dataset/state_transition_probabilities_granular.csv")
S = np.sort(transitions_df["From State"].unique())

A = ["AM", "B"]  # Actions: Annual Mammogram (AM) and Biopsy (B)
T = 60  # Time steps, e.g., from age 40 to 99

# Read the rewards from the CSV file into a pandas DataFrame
rewards_df = pd.read_csv("/Users/ruchithakor/Downloads/Masters_Docs/MRP/MRP_Optimal_Biopsy_Decision_Making_Breast_Cancer_RL/dataset/rewards.csv")

# Convert the DataFrame into a dictionary with the required format for the MDP
# The dictionary will be structured as R[t][state][action]
R = {row["age"]: {
        "Healthy": {"AM": row["Healthy_AM"], "B": row["Healthy_B"]},
        "Early": {"AM": row["Early_AM"], "B": row["Early_B"]},
        "Advanced": {"AM": row["Advanced_AM"], "B": row["Advanced_b"]}
    } for _, row in rewards_df.iterrows()}

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
    P[age][from_state]['AM'][to_state] = probability
    P[age][from_state]['B'][to_state] = 0 if to_state != 100 else 1 # If biopsy performed patient reaches state 100 and cycle closes

# Ensure all states have transition probabilities for all ages
for age in range(40, 100):
    for state in S:  # Assuming states are numbered 1 through 100
        if state not in P[age]:
            P[age][state] = {"AM": {state: 1}, "B": {100: 1}}  # Self-loop for 'AM', transition to 100 for 'B'
        else:
            if state != 100:  # State 100 is absorbing
                remaining_prob = 1 - sum(P[age][state]["AM"].values())
                P[age][state]["AM"][state] = remaining_prob  # Self-loop probability

# Initialize utilities for the final time step with terminal rewards
U = {100: {state: 1 if state in np.round(np.arange(0,21.1, 0.1), 1)  else 0 for state in S }}  # Assuming 0 terminal reward for all states except state 100
U[100][100] = -100  # Terminal reward for cancer detection


# Initialize policy
policy = {age: {state: None for state in S} for age in range(40, 100)}

# Backward induction with age-specific rewards and transition probabilities
for age in range(99, 39, -1):  # Backward from one year before the last age to the first age in the model
    U[age] = {}
    for state in S:
        if state == 100:
            # Absorbing state's utility remains the same for all time steps
            U[age][100] = U[100][100]
            policy[age][100] = "None"  # No action is taken at absorbing state
            continue

        # Compute the utility for each action
        action_utilities = {}

        for action in A:

            if action == "B":
                # If action is biopsy, transition to state 100
                if state in np.round(np.arange(0, 1, 0.1), 1): # healthy state
                    action_utilities[action] = float(R[age]["Healthy"][action]) 

                else:
                    # Early and advance stages of cancer varies with age
                    # assign rewards accordingly
                    if age in list(range(40,46)):
                        if state in np.round(np.arange(1, 5, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action])
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) 
                    
                    elif age in list(range(46, 66)):
                        if state in np.round(np.arange(1, 9, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) 
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action])
                    
                    elif age in list(range(66, 81)):
                        if state in np.round(np.arange(1, 11, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) 
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action])
                    
                    elif age in list(range(81, 91)):
                        if state in list(range(1, 16)):
                            action_utilities[action] = float(R[age]["Early"][action]) 
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) 
                    
                    elif age in list(range(91, 95)):
                        if state in list(range(1, 18)):
                            action_utilities[action] = float(R[age]["Early"][action]) 
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) 
                    
                    else:
                        if state in np.round(np.arange(1, 21, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) 
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) 
                    
            else:
                # For AM or other actions that don't lead to an absorbing state

                # calculate expected future utility for all possible next states
                expected_future_utility = sum(P[age][state][action].get(next_s, 0) * U[age + 1].get(next_s, 0) for next_s in S)
                if state in  np.round(np.arange(0, 1, 0.1), 1):  # healthy state
                    action_utilities[action] = float(R[age]["Healthy"][action]) + expected_future_utility
                else:
                    if age in list(range(40,46)):
                        if state in np.round(np.arange(1, 5, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) + expected_future_utility
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) +  0.05 * expected_future_utility
                    
                    elif age in list(range(46, 66)):
                        if state in np.round(np.arange(1, 9, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) + expected_future_utility
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) + 0.05 * expected_future_utility
                    
                    elif age in list(range(66, 81)):
                        if state in np.round(np.arange(1, 11, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) + expected_future_utility
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) + 0.05 * expected_future_utility
                    
                    elif age in list(range(81, 91)):
                        if state in np.round(np.arange(1, 16, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) + expected_future_utility
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) + 0.05 * expected_future_utility
                    
                    elif age in list(range(91, 95)):
                        if state in np.round(np.arange(1, 18, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) + expected_future_utility
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) + 0.05 * expected_future_utility
                    
                    else:
                        if state in np.round(np.arange(1, 21, 0.1), 1):
                            action_utilities[action] = float(R[age]["Early"][action]) + expected_future_utility
                        else:
                            action_utilities[action] = float(R[age]["Advanced"][action]) + 0.05 * expected_future_utility
                    
        # Select the action that maximizes the expected utility
        optimal_action = max(action_utilities, key = action_utilities.get)
        U[age][state] = action_utilities[optimal_action]
        policy[age][state] = optimal_action

# policy now contains the optimal action for each state and time step
# create a dataframe of policies
data = []
for age in range(40, 100):
    for state in S:
        data.append({
            "time_stamp": age, "state": state, "action": policy[age][state], "value": U[age][state]
        })

# Create a DataFrame from the list of tuples
df = pd.DataFrame(data)
# Save the transition probabilities to a CSV file
results_filename = "/Users/ruchithakor/Downloads/Masters_Docs/MRP/MRP_Optimal_Biopsy_Decision_Making_Breast_Cancer_RL/results/granular_level_results/backward_induction_policy_granular.csv"
df.to_csv(results_filename, index = False)

# Filter the DataFrame for rows where Action is 'B'
filtered_df = df[df["action"] == "B"]

# Group by 'Time Step' and get the first occurrence of 'B' for each time step
first_b_df = filtered_df.groupby("time_stamp").first().reset_index()

# Select the necessary columns
result_df = first_b_df[["time_stamp", "state"]]
b_r_filename = "/Users/ruchithakor/Downloads/Masters_Docs/MRP/MRP_Optimal_Biopsy_Decision_Making_Breast_Cancer_RL/results/granular_level_results/backward_induction_threshold_granular.csv"
result_df.to_csv(b_r_filename, index = False)