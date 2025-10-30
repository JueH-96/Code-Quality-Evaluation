import sys
from collections import defaultdict

n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))

positions = defaultdict(list)
for idx, color in enumerate(a):
    positions[color].append(idx + 1)  # Convert to 1-based index

count = 0
for color in positions:
    x, y = positions[color]
    if y - x == 2:
        count += 1

print(count)