"""This python file calculates transition probability of moving from one state(risk score S = 0, 1, 2, ..., 100) to other 
for every time stamp(age) 40-100
"""

# Import required packages
import pandas as pd
from pathlib import Path

# read the data from all women data csv file
cwd_path = str(Path.cwd())
df_women = pd.read_csv(cwd_path + "/dataset/all_women_data.csv")

# Initialize the transition counts and totals for each age
transition_counts = {age: {} for age in range(39, 100)}
transition_totals = {age: {} for age in range(39, 100)}

# Populate the counts and totals using data
for woman_id in df_women["Patient ID"].unique():

    # get the data of each woman
    woman_data = df_women[df_women["Patient ID"] == woman_id].sort_values("Age")

    # initialize default previous state and age
    previous_state = 0 
    previous_age = 39

    for index, row in woman_data.iterrows():
        # for all the available data for each woman at different age
        current_state = row["Risk Score"] # get current risk/probability of cancer
        current_age = row["Age"] # get current age
        
        if previous_state is not None and previous_age == current_age - 1:
        # check that previous state is not None and previous age is 1 year less than current
        
            if 40 <= current_age < 100:  # Ensure age is within the specified range
                # Increment the transition count
                if previous_state not in transition_counts[current_age]:
                    transition_counts[current_age][previous_state] = {}
                transition_counts[current_age][previous_state][current_state] = \
                    transition_counts[current_age][previous_state].get(current_state, 0) + 1
                # Increment the total count of transitions from the previous state at the current age
                transition_totals[current_age][previous_state] = \
                    transition_totals[current_age].get(previous_state, 0) + 1
        
        # set current age and state as previous age and state
        previous_state = current_state
        previous_age = current_age

# Calculate transition probabilities for each age
transition_probabilities = {age: {} for age in range(40, 100)}
for age in range(40, 100):
    for from_state, to_states in transition_counts[age].items():
        transition_probabilities[age][from_state] = {}
        for to_state, count in to_states.items():
            transition_probabilities[age][from_state][to_state] = count / transition_totals[age][from_state]

# Convert the transition probabilities to a DataFrame
transitions_list = []
for t in range(40, 100):
    for from_state, to_states in transition_probabilities[t].items():
        for to_state, prob in to_states.items():
            transitions_list.append({
                "Age": t,
                "From State": from_state,
                "To State": to_state,
                "Probability": prob
            })

transitions_df = pd.DataFrame(transitions_list)

# Format the DataFrame to avoid scientific notation, round to 4 decimal places.
for col in transitions_df.columns:
    if transitions_df[col].dtype == "float":
        transitions_df[col] = transitions_df[col].map(lambda x: "{:.5f}".format(x))

# Save the transition probabilities to a CSV file
transitions_csv_filename = cwd_path + "/dataset/state_transition_probabilities.csv"
transitions_df.to_csv(transitions_csv_filename, index=False)
