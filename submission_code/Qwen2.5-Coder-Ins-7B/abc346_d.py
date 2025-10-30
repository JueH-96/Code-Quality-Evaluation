# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]
C = list(map(int, data[2:]))

# Initialize the minimum cost to a large number
min_cost = float('inf')

# Iterate over each possible position i where the characters are the same
for i in range(1, N):
    if S[i] == S[i - 1]:
        # Calculate the cost of making the string good by changing the characters before and after position i
        cost = sum(C[:i]) + sum(C[i + 1:])
        # Update the minimum cost if the current cost is lower
        min_cost = min(min_cost, cost)

# Print the minimum cost
print(min_cost)