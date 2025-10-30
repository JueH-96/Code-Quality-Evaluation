import sys
from collections import deque

def bfs(grid, start):
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    queue = deque([start])
    visited[start[0]][start[1]] = True
    freedom = 0

    while queue:
        x, y = queue.popleft()
        freedom += 1

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == '.':
                visited[nx][ny] = True
                queue.append((nx, ny))

    return freedom

def max_freedom(grid):
    h, w = len(grid), len(grid[0])
    max_freedom = 0

    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                max_freedom = max(max_freedom, bfs(grid, (i, j)))

    return max_freedom

def main():
    input = sys.stdin.read
    data = input().split()

    H, W = int(data[0]), int(data[1])
    grid = data[2:2 + H]

    result = max_freedom(grid)
    print(result)

if __name__ == "__main__":
    main()