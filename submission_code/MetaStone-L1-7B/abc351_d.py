import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())
    
    is_free = [[False for _ in range(W)] for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '.':
                is_free[i][j] = False
                continue
            adj = False
            # Check up
            if i > 0 and grid[i-1][j] == '.':
                adj = True
            # Check down
            if i < H-1 and grid[i+1][j] == '.':
                adj = True
            # Check left
            if j > 0 and grid[i][j-1] == '.':
                adj = True
            # Check right
            if j < W-1 and grid[i][j+1] == '.':
                adj = True
            is_free[i][j] = not adj
    
    visited = [[False for _ in range(W)] for _ in range(H)]
    max_size = 0
    
    for i in range(H):
        for j in range(W):
            if is_free[i][j] and not visited[i][j]:
                size = 0
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                size += 1
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if is_free[nx][ny] and not visited[nx][ny]:
                                visited[nx][ny] = True
                                q.append((nx, ny))
                                size += 1
                if size > max_size:
                    max_size = size
    print(max_size)

if __name__ == "__main__":
    main()