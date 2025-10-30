import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    allowed = [[False] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                continue
            has_magnet = False
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                    has_magnet = True
                    break
            allowed[i][j] = not has_magnet
    
    visited_allowed = [[False] * W for _ in range(H)]
    components = []
    for i in range(H):
        for j in range(W):
            if allowed[i][j] and not visited_allowed[i][j]:
                q = deque()
                q.append((i, j))
                visited_allowed[i][j] = True
                component = []
                while q:
                    x, y = q.popleft()
                    component.append((x, y))
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and allowed[nx][ny] and not visited_allowed[nx][ny]:
                            visited_allowed[nx][ny] = True
                            q.append((nx, ny))
                components.append(component)
    
    max_size = 0
    for comp in components:
        visited = [[False] * W for _ in range(H)]
        q = deque()
        size = 0
        for (x, y) in comp:
            q.append((x, y))
            visited[x][y] = True
            size += 1
        while q:
            x, y = q.popleft()
            if allowed[x][y]:
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny]:
                        visited[nx][ny] = True
                        q.append((nx, ny))
                        size += 1
        if size > max_size:
            max_size = size
    print(max_size)

if __name__ == "__main__":
    main()