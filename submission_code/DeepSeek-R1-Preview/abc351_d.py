import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    is_free = [[False for _ in range(W)] for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                is_free[i][j] = False
                continue
            free = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni = i + dx
                nj = j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] == '#':
                        free = False
                        break
            is_free[i][j] = free
    
    visited = [[False for _ in range(W)] for _ in range(H)]
    max_degree = 1  # At least 1 since there's at least one non-magnet cell
    
    for i in range(H):
        for j in range(W):
            if visited[i][j] or not is_free[i][j]:
                continue
            # BFS to find component
            queue = deque()
            queue.append((i, j))
            visited[i][j] = True
            component = set()
            component.add((i, j))
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if not visited[nx][ny] and is_free[nx][ny]:
                            visited[nx][ny] = True
                            queue.append((nx, ny))
                            component.add((nx, ny))
            # Now, find adjacent non-component, non-magnet cells
            adjacent = set()
            for (x, y) in component:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if grid[nx][ny] != '#' and (nx, ny) not in component:
                            adjacent.add((nx, ny))
            # Compute degree
            degree = len(component) + len(adjacent)
            if degree > max_degree:
                max_degree = degree
    print(max_degree)

if __name__ == '__main__':
    main()