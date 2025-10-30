import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    stuck = [[False] * W for _ in range(H)]
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                for dx, dy in dirs:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            stuck[i][j] = True
                            break
    
    visited = [[False] * W for _ in range(H)]
    max_ans = 1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not stuck[i][j] and not visited[i][j]:
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                comp_size = 0
                stuck_set = set()
                
                while q:
                    x, y = q.popleft()
                    comp_size += 1
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                            if stuck[nx][ny]:
                                stuck_set.add((nx, ny))
                            else:
                                if not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    q.append((nx, ny))
                
                total = comp_size + len(stuck_set)
                if total > max_ans:
                    max_ans = total
                    
    print(max_ans)

if __name__ == '__main__':
    main()