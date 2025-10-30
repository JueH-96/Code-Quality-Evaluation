import sys
from collections import deque

def main():
    h, w = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    # Initialize movable grid
    movable = [[False]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#':
                continue
            has_magnet = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == '#':
                    has_magnet = True
                    break
            if not has_magnet:
                movable[i][j] = True
    
    # Find connected components of movable cells
    visited = [[False]*w for _ in range(h)]
    components = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#' or visited[i][j] or not movable[i][j]:
                continue
            q = deque()
            q.append((i, j))
            visited[i][j] = True
            component = [(i, j)]
            while q:
                x, y = q.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if not visited[nx][ny] and movable[nx][ny]:
                            visited[nx][ny] = True
                            component.append((nx, ny))
                            q.append((nx, ny))
            components.append(component)
    
    max_k = 1
    for comp in components:
        cells = set(comp)
        for x, y in comp:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    cells.add((nx, ny))
        k = len(cells)
        if k > max_k:
            max_k = k
    
    print(max_k)

if __name__ == "__main__":
    main()