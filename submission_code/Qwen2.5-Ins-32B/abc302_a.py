import math

# Read input from stdin
A, B = map(int, input().split())

# Calculate the minimum number of attacks needed
attacks_needed = math.ceil(A / B)

# Write the answer to stdout
print(attacks_needed)