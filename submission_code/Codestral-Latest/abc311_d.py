# YOUR CODE HERE
import sys
from collections import deque

def bfs(grid, start):
    n, m = len(grid), len(grid[0])
    visited = [[False] * m for _ in range(n)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque([start])
    visited[start[0]][start[1]] = True
    count = 1

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == '.':
                if not visited[nx][ny]:
                    visited[nx][ny] = True
                    count += 1
                    queue.append((nx, ny))
                nx += dx
                ny += dy

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    n = int(data[0])
    m = int(data[1])
    grid = data[2:]

    count = bfs(grid, (2, 2))
    print(count)

if __name__ == "__main__":
    main()