# YOUR CODE HERE
def min_cost_to_good_string(N, S, C):
    # To make the string a good string, we need exactly one pair of consecutive characters to be the same.
    # We will iterate through the string and calculate the cost to make each pair same.
    
    min_cost = float('inf')
    
    for i in range(N - 1):
        if S[i] != S[i + 1]:
            # Calculate the cost to make S[i] and S[i + 1] the same
            cost = min(C[i], C[i + 1])
            if cost < min_cost:
                min_cost = cost
    
    # If min_cost is still infinity, it means we cannot make any pair same, which is impossible given the constraints.
    return min_cost

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
C = list(map(int, data[2:]))

# Output the result
print(min_cost_to_good_string(N, S, C))