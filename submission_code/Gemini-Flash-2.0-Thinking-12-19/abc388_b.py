import sys

# Read N and D
N, D = map(int, sys.stdin.readline().split())

# Read snake initial thickness and length
snakes = []
for _ in range(N):
    T, L = map(int, sys.stdin.readline().split())
    snakes.append((T, L))

# For each integer k from 1 to D
for k in range(1, D + 1):
    max_weight_k = 0
    # Calculate the weight for each snake with increased length and find the maximum
    for T, L in snakes:
        current_length = L + k
        current_weight = T * current_length
        max_weight_k = max(max_weight_k, current_weight)
    
    # Print the maximum weight for day k
    print(max_weight_k)