def min_cost_to_make_good_string(N, S, costs):
    # Initialize the minimum cost to a large number
    min_cost = float('inf')

    # We will check for each possible position to create the good string
    for i in range(N - 1):
        # We want to create a good string with the same characters at position i and i+1
        # We will consider two cases:
        # 1. Make S[i] and S[i+1] both '0'
        # 2. Make S[i] and S[i+1] both '1'
        
        # Calculate the cost to make S[i] and S[i+1] both '0'
        cost_to_zero = 0
        if S[i] == '1':
            cost_to_zero += costs[i]
        if S[i + 1] == '1':
            cost_to_zero += costs[i + 1]
        
        # Calculate the cost to make S[i] and S[i+1] both '1'
        cost_to_one = 0
        if S[i] == '0':
            cost_to_one += costs[i]
        if S[i + 1] == '0':
            cost_to_one += costs[i + 1]
        
        # Update the minimum cost
        min_cost = min(min_cost, cost_to_zero, cost_to_one)

    return min_cost

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
S = data[1]
costs = list(map(int, data[2].split()))

# Get the result
result = min_cost_to_make_good_string(N, S, costs)

# Print the result
print(result)