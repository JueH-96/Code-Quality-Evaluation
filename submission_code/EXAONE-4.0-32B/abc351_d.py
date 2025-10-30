import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    free = [[True] * W for _ in range(H)]
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                free[i][j] = False
            else:
                for dx, dy in dirs:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            free[i][j] = False
                            break
    
    ans = 1
    visited = [[False] * W for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and free[i][j] and not visited[i][j]:
                comp_free = []
                blocked_set = set()
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                comp_free.append((i, j))
                
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if not visited[nx][ny] and grid[nx][ny] == '.' and free[nx][ny]:
                                visited[nx][ny] = True
                                comp_free.append((nx, ny))
                                queue.append((nx, ny))
                
                for (x, y) in comp_free:
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '.' and not free[nx][ny]:
                                blocked_set.add((nx, ny))
                
                total = len(comp_free) + len(blocked_set)
                if total > ans:
                    ans = total
                    
    print(ans)

if __name__ == '__main__':
    main()