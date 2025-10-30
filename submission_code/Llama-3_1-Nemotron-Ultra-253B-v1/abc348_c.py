import sys
from collections import defaultdict

n = int(sys.stdin.readline())
color_groups = defaultdict(list)

for _ in range(n):
    a, c = map(int, sys.stdin.readline().split())
    color_groups[c].append(a)

min_values = [min(group) for group in color_groups.values()]
print(max(min_values))