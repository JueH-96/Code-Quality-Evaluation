from collections import deque
import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    can_exit = [[False for _ in range(W)] for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                has_adj = False
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            has_adj = True
                            break
                can_exit[i][j] = not has_adj
    
    visited = [[False for _ in range(W)] for _ in range(H)]
    max_degree = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and can_exit[i][j] and not visited[i][j]:
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                count = 0
                while q:
                    x, y = q.popleft()
                    count += 1
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '.' and not visited[nx][ny]:
                                visited[nx][ny] = True
                                if can_exit[nx][ny]:
                                    q.append((nx, ny))
                if count > max_degree:
                    max_degree = count
    
    # Check cells that cannot exit (degree is 1)
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not can_exit[i][j]:
                if 1 > max_degree:
                    max_degree = 1
    
    print(max_degree)

if __name__ == '__main__':
    main()