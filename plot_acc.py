import pandas as pd

import matplotlib.pyplot as plt

def plot_accuracy_from_csv(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path, header=None)
    
    # Extract accuracy values
    accuracy_values = data[0].values
    
    # Generate x values (iterations)
    iterations = [(i+1) * 200 for i in range(len(accuracy_values))]
    
    # Plot the data
    plt.plot(iterations, accuracy_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iteration')
    plt.grid(True)
    plt.show()

# Example usage
plot_accuracy_from_csv('val_acc_history.csv')