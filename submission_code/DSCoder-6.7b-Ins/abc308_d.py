import sys
from collections import deque

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]

    snuke = 'snuke'
    visited = [[0]*W for _ in range(H)]
    queue = deque([(0, 0, 0)])

    while queue:
        x, y, t = queue.popleft()
        if (x, y) == (H-1, W-1):
            return 'Yes'
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and S[nx][ny] == snuke[(t+1)%5]:
                visited[nx][ny] = 1
                queue.append((nx, ny, t+1))
    return 'No'

print(solve())