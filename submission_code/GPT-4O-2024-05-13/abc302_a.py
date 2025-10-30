# YOUR CODE HERE
import math

# Read input
A, B = map(int, input().split())

# Calculate the minimum number of attacks
min_attacks = math.ceil(A / B)

# Print the result
print(min_attacks)