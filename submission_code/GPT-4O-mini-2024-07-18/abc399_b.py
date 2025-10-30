def determine_ranks(N, scores):
    ranks = [0] * N  # Initialize ranks array
    r = 1  # Start rank from 1
    undetermined = set(range(N))  # Set of indices with undetermined ranks

    while undetermined:
        # Find the maximum score among the undetermined ranks
        max_score = max(scores[i] for i in undetermined)
        # Find all indices with the maximum score
        max_indices = [i for i in undetermined if scores[i] == max_score]
        
        # Assign the current rank to all people with the maximum score
        for index in max_indices:
            ranks[index] = r
        
        # Update the rank for the next group
        r += len(max_indices)
        # Remove these indices from the undetermined set
        undetermined.difference_update(max_indices)

    return ranks

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
scores = list(map(int, data[1].split()))

# Get the ranks
ranks = determine_ranks(N, scores)

# Print the ranks
for rank in ranks:
    print(rank)