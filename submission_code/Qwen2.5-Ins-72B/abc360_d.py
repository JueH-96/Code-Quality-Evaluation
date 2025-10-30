# YOUR CODE HERE
import sys

def count_passing_ants(N, T, S, X):
    # Create a list of tuples (X_i, direction) for each ant
    ants = [(X[i], int(S[i])) for i in range(N)]
    
    # Sort the ants by their initial positions
    ants.sort()
    
    # Initialize the count of passing pairs
    passing_pairs = 0
    
    # Use a prefix sum array to count the number of ants facing right
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + ants[i][1]
    
    # Iterate through the sorted ants
    for i in range(N):
        if ants[i][1] == 0:  # If the ant is facing left
            # Count the number of ants facing right that are to the right of the current ant
            passing_pairs += prefix_sum[N] - prefix_sum[i + 1]
    
    return passing_pairs

# Read input
input = sys.stdin.read
data = input().split()
N, T = map(int, data[0].split())
S = data[1]
X = list(map(int, data[2:]))

# Solve the problem
result = count_passing_ants(N, T, S, X)

# Print the result
print(result)