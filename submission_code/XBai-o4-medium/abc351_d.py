import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Precompute is_safe grid
    is_safe = [[False] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                safe = True
                for dx, dy in directions:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            safe = False
                            break
                is_safe[i][j] = safe
            else:
                is_safe[i][j] = False
    
    visited = [[False] * W for _ in range(H)]
    max_degree = 1  # At least 1 for non-safe cells
    
    for i in range(H):
        for j in range(W):
            if is_safe[i][j] and not visited[i][j]:
                # BFS to find component size
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                component_size = 1
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if is_safe[nx][ny] and not visited[nx][ny]:
                                visited[nx][ny] = True
                                queue.append((nx, ny))
                                component_size += 1
                if component_size > max_degree:
                    max_degree = component_size
    
    print(max_degree)

if __name__ == "__main__":
    main()