# YOUR CODE HERE

import sys
from collections import deque

H, W = map(int, input().split())
grid = [input() for _ in range(H)]

def is_valid(i, j):
    return 0 <= i < H and 0 <= j < W

def bfs():
    queue = deque([(0, 0, 0)])
    visited = [[0]*W for _ in range(H)]
    visited[0][0] = 1
    while queue:
        i, j, k = queue.popleft()
        if i == H-1 and j == W-1:
            return True
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if is_valid(ni, nj) and visited[ni][nj] == 0 and grid[ni][nj] == "snuke"[(k+1)%5]:
                visited[ni][nj] = 1
                queue.append((ni, nj, (k+1)%5))
    return False

print("Yes" if bfs() else "No")