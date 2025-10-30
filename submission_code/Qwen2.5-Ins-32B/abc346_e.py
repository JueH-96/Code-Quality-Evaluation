import sys
from collections import defaultdict

input = sys.stdin.read
H, W, M, *queries = map(int, input().split())
row_color = [0] * H
col_color = [0] * W
color_count = defaultdict(int)

for T, A, X in zip(*[iter(queries)] * 3):
    if T == 1:
        row_color[A - 1] = X
    else:
        col_color[A - 1] = X

for i in range(H):
    for j in range(W):
        color = row_color[i] if row_color[i] > col_color[j] else col_color[j]
        color_count[color] += 1

color_count.pop(0, None)
colors = sorted(color_count.keys())
print(len(colors))
for color in colors:
    print(color, color_count[color])