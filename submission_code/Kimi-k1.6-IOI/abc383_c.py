import sys
from collections import deque

def main():
    h, w, d = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    h_positions = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 'H':
                h_positions.append((i, j))
    
    if not h_positions:
        print(0)
        return
    
    dist = [[-1] * w for _ in range(h)]
    q = deque()
    for i, j in h_positions:
        dist[i][j] = 0
        q.append((i, j))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while q:
        i, j = q.popleft()
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                if grid[ni][nj] != '#' and dist[ni][nj] == -1:
                    dist[ni][nj] = dist[i][j] + 1
                    q.append((ni, nj))
    
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] != '#' and dist[i][j] != -1 and dist[i][j] <= d:
                count += 1
    print(count)

if __name__ == "__main__":
    main()