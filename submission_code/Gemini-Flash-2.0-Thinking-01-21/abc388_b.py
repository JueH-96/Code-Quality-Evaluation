import sys

# Read N and D
line = sys.stdin.readline().split()
N = int(line[0])
D = int(line[1])

# Read snake data
snakes = []
for _ in range(N):
    line = sys.stdin.readline().split()
    T = int(line[0])
    L = int(line[1])
    snakes.append((T, L))

# Process for each k from 1 to D
for k in range(1, D + 1):
    max_weight_for_k = 0 # Initialize max weight for the current k

    # Calculate weights for each snake and find the maximum
    for T, L in snakes:
        current_weight = T * (L + k)
        max_weight_for_k = max(max_weight_for_k, current_weight)

    # Print the maximum weight for the current k
    print(max_weight_for_k)