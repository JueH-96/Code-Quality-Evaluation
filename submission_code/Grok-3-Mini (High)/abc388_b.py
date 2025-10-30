import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Read N and D from the first two elements
N = int(data[0])
D = int(data[1])

# The remaining data from index 2 onwards contains the snake data
snake_data = data[2:]

# Create a list of tuples for each snake's thickness and length
snakes = [(int(snake_data[i * 2]), int(snake_data[i * 2 + 1])) for i in range(N)]

# For each k from 1 to D, calculate the maximum weight and print it
for k in range(1, D + 1):
    max_weight = max(T * (L + k) for T, L in snakes)
    print(max_weight)