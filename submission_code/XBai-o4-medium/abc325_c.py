import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    visited = [[False for _ in range(W)] for _ in range(H)]
    count = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                count += 1
                # BFS to mark all connected sensors
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    # Check all 8 directions
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < H and 0 <= ny < W:
                                if grid[nx][ny] == '#' and not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    q.append((nx, ny))
    print(count)

if __name__ == "__main__":
    main()