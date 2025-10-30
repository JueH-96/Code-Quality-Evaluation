import sys

def max_weight_edges(N, weights):
    # Initialize the maximum weight to 0
    max_weight = 0

    # Iterate through all possible pairs of vertices
    for i in range(1, N):
        for j in range(i+1, N+1):
            # Add the weight of the edge connecting vertices i and j to the total weight
            max_weight += weights[i-1][j-1]

    return max_weight

# Read the input from stdin
N = int(sys.stdin.readline().strip())
weights = []
for i in range(N):
    row = [int(x) for x in sys.stdin.readline().strip().split()]
    weights.append(row)

# Compute the maximum weight of the chosen edges
result = max_weight_edges(N, weights)

# Print the result to stdout
print(result)