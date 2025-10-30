from collections import deque
from sys import stdin

def is_valid_move(grid, visited, x, y):
    """Check if a move to (x, y) is valid."""
    return (0 <= x < len(grid) and 0 <= y < len(grid[0]) and 
            grid[x][y] == '.' and not visited[x][y])

def has_adjacent_magnet(grid, x, y):
    """Check if there is a magnet adjacent to (x, y)."""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == '#':
            return True
    return False

def bfs(grid, x, y):
    """Perform BFS from (x, y) and return the number of reachable cells."""
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    queue = deque([(x, y)])
    visited[x][y] = True
    count = 1

    while queue:
        cx, cy = queue.popleft()
        if has_adjacent_magnet(grid, cx, cy):
            continue
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if is_valid_move(grid, visited, nx, ny):
                queue.append((nx, ny))
                visited[nx][ny] = True
                count += 1

    return count

def main():
    H, W = map(int, stdin.readline().split())
    grid = [list(stdin.readline().strip()) for _ in range(H)]

    max_degree = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                degree = bfs(grid, i, j)
                max_degree = max(max_degree, degree)

    print(max_degree)

if __name__ == "__main__":
    main()