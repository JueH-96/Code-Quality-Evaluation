N = int(input())
colors = list(map(int, input().split()))
positions = [[] for _ in range(N+1)]
for idx, color in enumerate(colors, start=1):
    positions[color].append(idx)
count = 0
for color in range(1, N+1):
    p1, p2 = positions[color]
    if p2 - p1 == 2:
        count += 1
print(count)