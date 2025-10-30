import sys
from collections import deque

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    dist = [[-1] * W for _ in range(H)]
    q = deque()
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                q.append((i, j))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS to find shortest distance from any H
    while q:
        i, j = q.popleft()
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#' or dist[ni][nj] != -1:
                    continue
                dist[ni][nj] = dist[i][j] + 1
                q.append((ni, nj))
    
    # Count all non-wall cells with distance <= D and reachable
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                continue
            if dist[i][j] != -1 and dist[i][j] <= D:
                count += 1
    print(count)

if __name__ == "__main__":
    main()