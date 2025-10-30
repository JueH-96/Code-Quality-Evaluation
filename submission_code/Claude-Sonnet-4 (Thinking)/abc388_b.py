# YOUR CODE HERE
# Read input
n, d = map(int, input().split())
snakes = []
for _ in range(n):
    t, l = map(int, input().split())
    snakes.append((t, l))

# For each k from 1 to d
for k in range(1, d + 1):
    max_weight = 0
    for t, l in snakes:
        weight = t * (l + k)
        max_weight = max(max_weight, weight)
    print(max_weight)