import sys
import math

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse the input values
A = int(data[0])
B = int(data[1])

# Calculate the number of attacks needed
attacks = math.ceil(A / B)

# Print the result
print(attacks)