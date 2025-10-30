from sys import stdin

def dfs(grid, visited, i, j):
    """Perform a depth-first search from the given cell."""
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '#' or visited[i][j]:
        return
    visited[i][j] = True
    for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        dfs(grid, visited, i + x, j + y)

def count_sensors(grid):
    """Count the number of sensors in the grid."""
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '#' and not visited[i][j]:
                dfs(grid, visited, i, j)
                count += 1
    return count

def main():
    """Read input and print the answer."""
    H, W = map(int, stdin.readline().split())
    grid = [list(stdin.readline().strip()) for _ in range(H)]
    print(count_sensors(grid))

if __name__ == "__main__":
    main()