#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import time

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_data = pd.read_csv("train_data.csv")
    val_data = pd.read_csv("validation_data.csv")
    return train_data, val_data

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    DAG = bn.structure_learning.fit(df)
    model = bn.parameter_learning.fit(DAG, df)
    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Step 1: Create the initial Bayesian Network structure
    DAG = bn.structure_learning.fit(df)
    
    # Step 2: Prune edges based on statistical significance (e.g., chi-squared test)
    adjmat = DAG['adjmat']
    edges_to_remove = []

    for parent in adjmat.index:
        for child in adjmat.columns:
            if adjmat.at[parent, child] == 1:  # Check if an edge exists
                contingency_table = pd.crosstab(df[parent], df[child])
                _, p_value, _, _ = chi2_contingency(contingency_table)
                
                # Prune the edge if the p-value is above a chosen significance threshold (e.g., 0.05)
                if p_value > 0.05:
                    edges_to_remove.append((parent, child))

    # Step 3: Create a pruned edge list
    edges = [(parent, child) for parent in adjmat.index for child in adjmat.columns if adjmat.at[parent, child] == 1]
    for parent, child in edges_to_remove:
        if (parent, child) in edges:
            edges.remove((parent, child))

    # Step 4: Construct the pruned Bayesian network structure with bnlearn
    pruned_DAG = bn.make_DAG(edges)

    # Step 5: Fit the pruned model with parameter learning
    pruned_model = bn.parameter_learning.fit(pruned_DAG, df)

    return pruned_model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    optimized_DAG = bn.structure_learning.fit(df, methodtype='hc')  # Hill Climbing
    optimized_model = bn.parameter_learning.fit(optimized_DAG, df)
    return optimized_model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

def plot_network(model, title="Bayesian Network"):
    """Visualize a Bayesian Network."""
    bn.plot(model)

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()
    
    start_time1 = time.time()
    # Create and save base model
    base_model = make_network(train_df)
    end_time1 = time.time()
    save_model("base_model.pkl", base_model)
    plot_network(base_model, "Initial Bayesian Network")
    
    
    
    # Create and save pruned model
    # pruned_network = make_pruned_network(train_df.copy())
    start_time = time.time()
    pruned_network = make_pruned_network(train_df)
    end_time = time.time()
    # print(f"Runtime for pruned model construction: {end_time - start_time} seconds")

    save_model("pruned_model.pkl", pruned_network)
    plot_network(pruned_network, "Pruned Bayesian Network")
    start_time3 = time.time()
    # Create and save optimized model
    optimized_network = make_optimized_network(train_df)
    end_time3 = time.time()
    save_model("optimized_model.pkl", optimized_network)
    plot_network(optimized_network, "Optimized Bayesian Network")

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)
    
    # print(f"Runtime for model construction: {end_time1 - start_time1} seconds")
    # print(f"Runtime for pruned model construction: {end_time - start_time} seconds")
    # print(f"Runtime for optimized model construction: {end_time3 - start_time3} seconds")

    print("[+] Done")

if __name__ == "__main__":
    main()

