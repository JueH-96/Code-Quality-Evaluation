import sys
from collections import deque

def main():
    h, w = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(h)]
    visited = [[False for _ in range(w)] for _ in range(h)]
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#' and not visited[i][j]:
                count += 1
                # BFS to mark all connected components
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx][ny] == '#' and not visited[nx][ny]:
                                visited[nx][ny] = True
                                q.append((nx, ny))
    print(count)

if __name__ == "__main__":
    main()