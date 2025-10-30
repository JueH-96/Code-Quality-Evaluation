import sys
from collections import deque

def main():
    h, w = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(h)]
    
    # Precompute whether each cell is a 'free' cell (no adjacent magnets)
    free = [[False] * w for _ in range(h)]
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] != '.':
                continue
            has_magnet = False
            for dx, dy in dirs:
                ni, nj = i + dx, j + dy
                if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == '#':
                    has_magnet = True
                    break
            if not has_magnet:
                free[i][j] = True
    
    # BFS to find connected components of free cells and calculate max degree
    component_id = [[-1] * w for _ in range(h)]
    comp_num = 0
    max_deg = 0
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.' and free[i][j] and component_id[i][j] == -1:
                # Start BFS for new component
                q = deque()
                q.append((i, j))
                component_id[i][j] = comp_num
                comp_num += 1
                F = 0
                A = set()
                
                while q:
                    x, y = q.popleft()
                    F += 1
                    # Check all four directions for adjacent cells
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx][ny] == '.' and not free[nx][ny]:
                                A.add((nx, ny))
                
                current_deg = F + len(A)
                if current_deg > max_deg:
                    max_deg = current_deg
    
    # If no free components, all empty cells have degree 1 (at least one exists)
    print(max_deg if max_deg != 0 else 1)

if __name__ == '__main__':
    main()