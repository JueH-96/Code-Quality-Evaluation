import sys
import math

# Read all input and convert to integers
data = list(map(int, sys.stdin.read().split()))

# Extract N, D, P
N = data[0]
D = data[1]
P = data[2]

# Extract the fare list F
F = data[3:3 + N]

# Sort the fares in ascending order
sorted_F = sorted(F)

# Compute prefix sums of the sorted fares
prefix_sum = [0]
cum_sum = 0
for fare in sorted_F:
    cum_sum += fare
    prefix_sum.append(cum_sum)

# Calculate floor(N/D) and ceil(N/D)
floor_N_D = N // D
ceil_N_D = (N + D - 1) // D

# Initialize minimum cost to infinity
min_cost = float('inf')

# Iterate over k from 0 to floor(N/D) inclusive
for k in range(0, floor_N_D + 1):
    X = N - k * D  # Number of uncovered days
    cost = P * k + prefix_sum[X]
    if cost < min_cost:
        min_cost = cost

# Consider the cost when k = ceil(N/D)
cost_ceiling = P * ceil_N_D
if cost_ceiling < min_cost:
    min_cost = cost_ceiling

# Output the minimum cost
print(min_cost)