# Read input
N, D = map(int, input().split())
snakes = [tuple(map(int, input().split())) for _ in range(N)]

# For each k from 1 to D
for k in range(1, D + 1):
    # Calculate the maximum weight for this k
    max_weight = max(t * (l + k) for t, l in snakes)
    print(max_weight)