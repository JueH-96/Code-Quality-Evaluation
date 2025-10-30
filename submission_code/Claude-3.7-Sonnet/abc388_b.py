# Read N and D
N, D = map(int, input().split())

# Read thickness and length of each snake
snakes = []
for _ in range(N):
    T, L = map(int, input().split())
    snakes.append((T, L))

# For each k from 1 to D
for k in range(1, D + 1):
    # Compute the weight of each snake and find the maximum
    max_weight = max(T * (L + k) for T, L in snakes)
    print(max_weight)