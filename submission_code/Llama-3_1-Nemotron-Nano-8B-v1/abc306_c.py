n = int(input())
a = list(map(int, input().split()))
from collections import defaultdict

positions = defaultdict(list)
for idx, num in enumerate(a):
    positions[num].append(idx + 1)  # Using 1-based index

result = []
for i in range(1, n + 1):
    mid = positions[i][1]
    result.append((mid, i))

result.sort()
print(' '.join(str(x[1]) for x in result))