import sys

# Read input from stdin
R, G, B = map(int, input().split())
C = input().strip()

# Determine the minimum amount of money Takahashi needs to buy one pen
if C == "Red":
    min_cost = min(G, B)
elif C == "Green":
    min_cost = min(R, B)
else:
    min_cost = min(R, G)

# Print the minimum amount of money
print(min_cost)