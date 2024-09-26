import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(rounds, loss1, loss_fed_avg, use_parallel):
    min_length = min(len(loss_fed_avg), rounds)
    epochs_range = range(1, min_length + 1)

    plt.plot(epochs_range, loss1[:min_length], 'g', label='Federated Q loss parallel' if use_parallel else 'Federated Q loss')
    plt.plot(epochs_range, loss_fed_avg[:min_length], 'b', label='Federated avg loss')
    plt.title('Loss')
    plt.xlabel('Communication rounds')
    plt.ylabel('Federated Q vs Federated Average loss')
    plt.legend()
    plt.savefig('federated_loss_plot.png')

def plotter1(dir_path, experiment_name):
    # Load the saved metrics
    accuracy_per_class_over_rounds = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/accuracy_per_class_over_rounds.npy"))
    precision_per_class_over_rounds = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/precision_per_class_over_rounds.npy"))
    recall_per_class_over_rounds = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/recall_per_class_over_rounds.npy"))
    f1_per_class_over_rounds = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/f1_per_class_over_rounds.npy"))
    round_numbers = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/round_numbers.npy"))

    # Determine the number of classes
    num_classes = accuracy_per_class_over_rounds.shape[1]

    # Create a directory to save the images
    output_dir = f"{experiment_name}/metrics_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the metrics for each class and save the plots
    for class_idx in range(num_classes):
        plt.figure(figsize=(10, 6))
        
        # Plot precision per class
        plt.plot(round_numbers, precision_per_class_over_rounds[:, class_idx], label='Precision', marker='o')
        
        # Plot recall per class
        plt.plot(round_numbers, recall_per_class_over_rounds[:, class_idx], label='Recall', marker='o')
        
        # Plot F1 score per class
        plt.plot(round_numbers, f1_per_class_over_rounds[:, class_idx], label='F1 Score', marker='o')
        
        plt.xlabel('Communication Rounds')
        plt.ylabel('Metric Value')
        plt.title(f'Metrics Over Communication Rounds for Class {class_idx}')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        output_path = os.path.join(output_dir, f'class_{class_idx}_metrics.png')
        plt.savefig(output_path)
        
        # Show the plot
        # plt.show()


def plotter2(dir_path, experiment_name):
    # Load the saved metrics
    accuracy_per_class_over_rounds = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/accuracy_per_class_over_rounds.npy"))
    round_numbers = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/round_numbers.npy"))

    # Determine the number of classes
    num_classes = accuracy_per_class_over_rounds.shape[1]

    # Create a directory to save the images
    output_dir = f"{experiment_name}/metrics_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the metrics for each class and save the plots
    for class_idx in range(num_classes):
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy per class
        plt.plot(round_numbers, accuracy_per_class_over_rounds[:, class_idx], label='Accuracy', marker='o')
        
        plt.xlabel('Communication Rounds')
        plt.ylabel('Metric Value')
        plt.title(f'Metrics Over Communication Rounds for Class {class_idx}')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        output_path = os.path.join(output_dir, f'class_accuracy_{class_idx}_metrics.png')
        plt.savefig(output_path)
        
        # Show the plot
        # plt.show()

def plotter3(dir_path, valid_dist, experiment_name):
    # Load the accuracy per class over rounds
    accuracy_per_class_over_rounds = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/accuracy_per_class_over_rounds.npy"))

    # Class distribution from the Counter
    class_distribution = dict(valid_dist)

    # Calculate the total number of samples
    total_samples = sum(class_distribution.values())

    # Calculate the weights for each class
    class_weights = {cls: count / total_samples for cls, count in class_distribution.items()}

    # Initialize an array to store the average weighted accuracy for each round
    avg_weighted_accuracies = []

    # Calculate the weighted accuracy for each round
    for round_idx in range(accuracy_per_class_over_rounds.shape[0]):
        accuracy_current_round = accuracy_per_class_over_rounds[round_idx]
        weighted_accuracy = sum(accuracy_current_round[cls] * class_weights[cls] for cls in class_distribution)
        avg_weighted_accuracies.append(weighted_accuracy)

    # Convert to numpy array for easier plotting
    avg_weighted_accuracies = np.array(avg_weighted_accuracies)

    output_dir = f"{experiment_name}/metrics_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot the average weighted accuracies over rounds
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_weighted_accuracies) + 1), avg_weighted_accuracies, marker='o')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Weighted Accuracy')
    plt.title('Average Weighted Accuracy Over Communication Rounds')
    plt.grid(True)

    # Save the figure
    output_path = os.path.join(output_dir, f'overall_accuracy.png')
    plt.savefig(output_path)

    # plt.show()

def plotter4(dir_path, smallest_k_ids, experiment_name):
    # Load the confusion matrices
    confusion_matrices = np.load(os.path.join(dir_path, f"{experiment_name}/npy results/confusion_matrices.npy"))

    # Indices for specific classes you are interested in
    specific_classes = smallest_k_ids

    # Initialize lists to store accuracies
    average_accuracies = []
    average_specific_class_accuracies = []

    # Calculate accuracies
    for cm in confusion_matrices:
        # Overall average accuracy
        overall_accuracy = np.trace(cm) / np.sum(cm)
        average_accuracies.append(overall_accuracy)

        # Accuracies for specific classes
        numerator = 0
        denominator = 0
        
        for cls in specific_classes:
            numerator += cm[cls, cls]
            denominator += np.sum(cm[cls, :])
        
        # Calculate the average accuracy for specific classes for this round
        average_accuracy_this_round = numerator / denominator if denominator != 0 else 0
        average_specific_class_accuracies.append(average_accuracy_this_round)

    # Rounds (assuming one matrix per round)
    rounds = list(range(1, len(confusion_matrices) + 1))

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(rounds, average_accuracies, label='Average Accuracy Over All Classes', marker='o', linestyle='-', color='blue')
    plt.plot(rounds, average_specific_class_accuracies, label=f'Average Accuracy of Specific Classes {str(smallest_k_ids)}', marker='x', linestyle='--', color='red')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Average Accuracies Over Rounds')
    plt.legend()
    plt.grid(True)

    output_dir = f"{experiment_name}/metrics_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    output_path = os.path.join(output_dir, f'last_k_comparision.png')
    plt.savefig(output_path)

    # plt.show()

def plotter5(list1, list2, experiment_name):

    # Create a plot
    plt.figure()
    plt.plot(list1, list2, marker='o', linestyle='-', color='b')  # Plot with markers and lines

    # Add labels and title
    plt.xlabel("Rounds")
    plt.ylabel("Divergence")
    plt.title("Rounds vs Divergence Plot")

    output_dir = f"{experiment_name}/metrics_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    output_path = os.path.join(output_dir, f'divergence_plot.png')
    plt.savefig(output_path)

    # Show the plot
    # plt.show()