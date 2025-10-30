from collections import deque
import sys

def main():
    h, w, d = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    dist = [[float('inf')] * w for _ in range(h)]
    visited = [[False] * w for _ in range(h)]
    q = deque()
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                visited[i][j] = True
                q.append((i, j))
    
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while q:
        i, j = q.popleft()
        current_d = dist[i][j]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                if grid[ni][nj] != '#' and not visited[ni][nj]:
                    new_d = current_d + 1
                    if new_d <= d:
                        dist[ni][nj] = new_d
                        visited[ni][nj] = True
                        q.append((ni, nj))
    
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] in ('.', 'H') and dist[i][j] <= d:
                count += 1
    print(count)

if __name__ == "__main__":
    main()