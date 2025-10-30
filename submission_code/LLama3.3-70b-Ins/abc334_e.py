from collections import deque
import sys

def bfs(grid, visited, i, j):
    """Perform BFS traversal from cell (i, j)"""
    queue = deque([(i, j)])
    visited[i][j] = True
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == '#' and not visited[nx][ny]:
                queue.append((nx, ny))
                visited[nx][ny] = True

def count_components(grid):
    """Count the number of connected components in the grid"""
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '#' and not visited[i][j]:
                bfs(grid, visited, i, j)
                count += 1
    return count

def main():
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    MOD = 998244353

    # Count the number of red cells
    red_cells = sum(row.count('.') for row in grid)

    # Initialize the total number of components
    total_components = 0

    # Iterate over each red cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Create a copy of the grid with the current red cell repainted green
                new_grid = [row[:] for row in grid]
                new_grid[i][j] = '#'

                # Count the number of components in the new grid
                components = count_components(new_grid)

                # Add the number of components to the total
                total_components += components

    # Calculate the expected value
    expected_value = (total_components * pow(red_cells, MOD - 2, MOD)) % MOD

    print(expected_value)

if __name__ == '__main__':
    main()