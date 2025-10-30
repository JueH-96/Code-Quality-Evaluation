n = int(input())
a = list(map(int, input().split()))

positions = {}

for idx, color in enumerate(a):
    if color not in positions:
        positions[color] = []
    positions[color].append(idx)

count = 0
for color in range(1, n + 1):
    first, second = positions[color]
    if second - first == 2:
        count += 1

print(count)