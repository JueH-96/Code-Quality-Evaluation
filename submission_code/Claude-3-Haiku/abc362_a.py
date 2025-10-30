import sys

# Read input
R, G, B = map(int, input().split())
C = input()

# Determine the minimum cost
if C == 'Red':
    min_cost = min(G, B)
elif C == 'Green':
    min_cost = min(R, B)
else:
    min_cost = min(R, G)

# Print the result
print(min_cost)