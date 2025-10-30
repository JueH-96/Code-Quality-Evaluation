import sys
from collections import defaultdict

n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
positions = defaultdict(list)
for idx, num in enumerate(a):
    positions[num].append(idx + 1)

result = sorted(range(1, n + 1), key=lambda x: positions[x][1])

print(' '.join(map(str, result)))