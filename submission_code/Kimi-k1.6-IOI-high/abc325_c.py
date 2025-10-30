from collections import deque
import sys

def main():
    h, w = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    visited = [[False for _ in range(w)] for _ in range(h)]
    count = 0
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#' and not visited[i][j]:
                count += 1
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx][ny] == '#' and not visited[nx][ny]:
                                visited[nx][ny] = True
                                queue.append((nx, ny))
    print(count)

if __name__ == "__main__":
    main()