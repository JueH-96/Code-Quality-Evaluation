import math

# Read the input from stdin
A, B = map(int, input().split())

# Calculate the number of attacks needed
attacks = math.ceil(A / B)

# Print the answer to stdout
print(attacks)