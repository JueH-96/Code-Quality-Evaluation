import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
M = int(data[2])

index = 3
operations = []
for _ in range(M):
    T = int(data[index])
    A = int(data[index + 1])
    X = int(data[index + 2])
    operations.append((T, A, X))
    index += 3

grid = [[0] * W for _ in range(H)]
color_count = defaultdict(int)

for T, A, X in operations:
    if T == 1:
        for j in range(W):
            grid[A - 1][j] = X
    elif T == 2:
        for i in range(H):
            grid[i][A - 1] = X

for row in grid:
    for cell in row:
        color_count[cell] += 1

colors = sorted(color_count.keys())
K = len(colors)

print(K)
for color in colors:
    print(color, color_count[color])