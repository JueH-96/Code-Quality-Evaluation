# Read input
N, D = map(int, input().split())
snakes = []
for _ in range(N):
    t, l = map(int, input().split())
    snakes.append((t, l))

# For each increase k from 1 to D
for k in range(1, D + 1):
    # Calculate maximum weight when length increases by k
    max_weight = 0
    for thickness, length in snakes:
        # New weight is thickness * (length + k)
        weight = thickness * (length + k)
        max_weight = max(max_weight, weight)
    
    # Print the maximum weight for this k
    print(max_weight)