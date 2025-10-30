import sys
from collections import deque

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    INF = 10**18
    dist = [[INF for _ in range(W)] for _ in range(H)]
    q = deque()
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                q.append((i, j))
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while q:
        i, j = q.popleft()
        for di, dj in dirs:
            x = i + di
            y = j + dj
            if 0 <= x < H and 0 <= y < W:
                if grid[x][y] != '#':
                    if dist[x][y] > dist[i][j] + 1:
                        dist[x][y] = dist[i][j] + 1
                        q.append((x, y))
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                count += 1
            elif grid[i][j] == '.' and dist[i][j] <= D:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()