# YOUR CODE HERE
def minimum_cost_to_good_string(N, S, C):
    # Initialize the minimum cost to a large number
    min_cost = float('inf')
    
    # Iterate through the string to find pairs of different consecutive characters
    for i in range(N - 1):
        if S[i] != S[i + 1]:
            # Calculate the cost to make S[i] and S[i+1] the same
            cost = min(C[i], C[i + 1])
            # Update the minimum cost
            min_cost = min(min_cost, cost)
    
    # Return the minimum cost found
    return min_cost

# Read input
import sys
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
S = data[1]
C = list(map(int, data[2:]))

# Calculate and print the minimum cost
print(minimum_cost_to_good_string(N, S, C))