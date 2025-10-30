n = int(input())
a = list(map(int, input().split()))

from collections import defaultdict
color_positions = defaultdict(list)

for idx, color in enumerate(a):
    color_positions[color].append(idx)

count = 0
for color in color_positions:
    positions = sorted(color_positions[color])
    if positions[1] - positions[0] == 2:
        count += 1

print(count)