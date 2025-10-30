# YOUR CODE HERE
from collections import defaultdict

H, W, M = map(int, input().split())

rows = defaultdict(int)
cols = defaultdict(int)

for _ in range(M):
    t, a, x = map(int, input().split())
    if t == 1:
        rows[a] = x
    else:
        cols[a] = x

color_counts = defaultdict(int)

for x in rows.values():
    color_counts[x] += W

for x in cols.values():
    color_counts[x] += H - len(rows)

print(len(color_counts))

for c, x in sorted(color_counts.items()):
    print(c, x)