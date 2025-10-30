import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Initialize movable array
    movable = [[False for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                movable[i][j] = False
            else:
                movable[i][j] = True
                # Check all four directions for magnets
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            movable[i][j] = False
                            break

    # Find components using BFS
    component_id = [[0] * W for _ in range(H)]
    components = []
    component_sizes = []
    current_component = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(H):
        for j in range(W):
            if movable[i][j] and component_id[i][j] == 0:
                current_component += 1
                q = deque()
                q.append((i, j))
                component_id[i][j] = current_component
                cells = []
                cells.append((i, j))
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if movable[nx][ny] and component_id[nx][ny] == 0:
                                component_id[nx][ny] = current_component
                                q.append((nx, ny))
                                cells.append((nx, ny))
                components.append(cells)
                component_sizes.append(len(cells))

    max_val = 1  # At least 1 for non-magnet cells

    for c in range(len(components)):
        cells = components[c]
        size = component_sizes[c]
        adj = set()
        for (x, y) in cells:
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] == '.' and component_id[nx][ny] != (c + 1):
                        adj.add((nx, ny))
        count = len(adj)
        total = size + count
        if total > max_val:
            max_val = total

    print(max_val)

if __name__ == "__main__":
    main()