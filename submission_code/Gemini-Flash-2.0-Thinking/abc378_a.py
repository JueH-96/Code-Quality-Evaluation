balls = list(map(int, input().split()))
counts = {}
for ball in balls:
    counts[ball] = counts.get(ball, 0) + 1

operations = 0
for color in counts:
    operations += counts[color] // 2

print(operations)