n, d = map(int, input().split())
snakes = [tuple(map(int, input().split())) for _ in range(n)]

for k in range(1, d + 1):
    max_weight = max(t * (l + k) for t, l in snakes)
    print(max_weight)