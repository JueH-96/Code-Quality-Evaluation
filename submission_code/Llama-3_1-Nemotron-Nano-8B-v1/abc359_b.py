n = int(input())
a = list(map(int, input().split()))
from collections import defaultdict

positions = defaultdict(list)
for idx, num in enumerate(a):
    positions[num].append(idx + 1)

count = 0
for color in positions:
    pos1, pos2 = positions[color]
    if abs(pos1 - pos2) == 2:
        count += 1

print(count)