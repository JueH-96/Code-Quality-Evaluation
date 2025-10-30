# YOUR CODE HERE
import sys
sys.setrecursionlimit(1 << 25)
H, W, N = map(int, sys.stdin.readline().split())

hole_set = set()
for _ in range(N):
    a, b = map(int, sys.stdin.readline().split())
    hole_set.add((a - 1, b - 1))

grid = [[1]*W for _ in range(H)]
for i, j in hole_set:
    grid[i][j] = 0

S = [[0]*W for _ in range(H)]
total = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == 1:
            if i == 0 or j == 0:
                S[i][j] = 1
            else:
                S[i][j] = min(S[i-1][j-1], S[i-1][j], S[i][j-1]) + 1
            total += S[i][j]
        else:
            S[i][j] = 0

print(total)