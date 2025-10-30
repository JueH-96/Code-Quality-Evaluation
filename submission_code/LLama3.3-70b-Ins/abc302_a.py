import math

# Read input from stdin
A, B = map(int, input().split())

# Calculate the minimum number of attacks needed
attacks = math.ceil(A / B)

print(attacks)