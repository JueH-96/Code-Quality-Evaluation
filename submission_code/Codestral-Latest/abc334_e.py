import sys
from collections import deque

MOD = 998244353

def bfs(grid, visited, start):
    queue = deque([start])
    visited[start[0]][start[1]] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and not visited[nx][ny] and grid[nx][ny] == '#':
                visited[nx][ny] = True
                queue.append((nx, ny))

def count_components(grid):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    components = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '#' and not visited[i][j]:
                bfs(grid, visited, (i, j))
                components += 1
    return components

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    grid = [list(data[i]) for i in range(2, H + 2)]

    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))

    total_components = 0
    for i, j in red_cells:
        grid[i][j] = '#'
        total_components += count_components(grid)
        grid[i][j] = '.'

    expected_value = total_components // len(red_cells)
    print(expected_value % MOD)

if __name__ == "__main__":
    main()