import math

def calculate_total_cost(N):
    total_cost = 0
    while N > 1:
        total_cost += N
        N = math.floor(N / 2)
    return total_cost

# Read input
N = int(input().strip())

# Calculate and print the total cost
print(calculate_total_cost(N))