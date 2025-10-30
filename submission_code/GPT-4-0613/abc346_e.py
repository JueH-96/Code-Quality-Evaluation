import sys
from collections import defaultdict

H, W, M = map(int, sys.stdin.readline().split())
grid = [[0]*W for _ in range(H)]
color_count = defaultdict(int)
color_count[0] = H*W

for _ in range(M):
    T, A, X = map(int, sys.stdin.readline().split())
    A -= 1
    if T == 1:
        for j in range(W):
            if grid[A][j] != X:
                color_count[grid[A][j]] -= 1
                grid[A][j] = X
                color_count[X] += 1
    else:
        for i in range(H):
            if grid[i][A] != X:
                color_count[grid[i][A]] -= 1
                grid[i][A] = X
                color_count[X] += 1

colors = list(color_count.items())
colors.sort()
print(len(colors))
for color, count in colors:
    print(color, count)