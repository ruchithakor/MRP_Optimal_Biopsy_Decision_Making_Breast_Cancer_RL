# Optimal Biopsy Decision Making for Breast Cancer using Reinforcement Learning

This repository contains the implementation of the project "Optimal Biopsy Decision Making for Breast Cancer using Reinforcement Learning (RL)." The project explores two primary RL methods: Backward Induction and Q-Learning, with an aim to generate optimal biopsy decision policies for different stages of breast cancer.

## Project Phases

### 1. Initial Setup
Implemented a Backward Induction algorithm as referenced in the literature review to generate policies for various stages (probability of cancer) at each timestamp (age), starting from 40 to 100 years.

### 2. Q-Learning
Implemented an advanced RL method, Q-Learning, to generate policies for each age-stage combination and compared the results with those from Backward Induction.

### 3. Testing Both Policies
Tested both policies via two methods:
- **Simulation Testing**: Simulated scenarios to evaluate the performance of the policies.
- **RSNA Screening Data**: Tested policies on RSNA screening data from Kaggle.

### 4. Generating Granular Policies
Generated policies at a more granular level with risk scores ranging from 0 to 100 in increments of 0.1 using both RL methods to observe improvements.

## Simulation Testing Details
In the simulation testing phase, policies were evaluated based on their performance in simulated environments. Mean and standard deviation (std) of rewards were calculated to measure the policies' effectiveness and stability.

## Results
Descriptive statistics and detailed data analysis of the working dataset were provided. The results section covers the comprehensive performance of the implemented RL methods, including mean rewards and standard deviations.

## Discussion
The discussion section highlights the scope for future research, particularly focusing on DQN techniques. The potential for using large clinical datasets and CAD models employed by radiologists for accurate testing and results is also discussed.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: numpy, pandas, etc.

### Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/ruchithakor/MRP_Optimal_Biopsy_Decision_Making_Breast_Cancer_RL.git
cd MRP_Optimal_Biopsy_Decision_Making_Breast_Cancer_RL
