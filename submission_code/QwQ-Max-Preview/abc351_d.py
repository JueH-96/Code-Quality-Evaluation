import sys
from collections import deque

def main():
    h, w = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    # Precompute whether each cell is movable
    is_movable = [[False for _ in range(w)] for _ in range(h)]
    total_non_magnet = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                total_non_magnet += 1
                can_move = True
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid[ni][nj] == '#':
                            can_move = False
                            break
                is_movable[i][j] = can_move
    
    visited = [[False for _ in range(w)] for _ in range(h)]
    max_reachable = 0
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.' and not visited[i][j] and is_movable[i][j]:
                # BFS to find the connected component of movable cells
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                component = [(i, j)]
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx][ny] == '.' and not visited[nx][ny] and is_movable[nx][ny]:
                                visited[nx][ny] = True
                                component.append((nx, ny))
                                q.append((nx, ny))
                # Collect adjacent non-component non-magnet cells
                adjacent = set()
                for (x, y) in component:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx][ny] == '.' and not visited[nx][ny]:
                                adjacent.add((nx, ny))
                current_size = len(component) + len(adjacent)
                if current_size > max_reachable:
                    max_reachable = current_size
    
    non_movable_non_magnet = total_non_magnet - sum(sum(row) for row in is_movable)
    if non_movable_non_magnet > 0:
        answer = max(max_reachable, 1)
    else:
        answer = max_reachable
    
    print(answer)

if __name__ == "__main__":
    main()