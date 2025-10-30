# YOUR CODE HERE

import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Initialize the visited array and the stack
    visited = [[False]*M for _ in range(N)]
    stack = [(2, 2)]
    visited[2][2] = True

    # Define the directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Depth-first search
    while stack:
        x, y = stack.pop()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] == '.' and not visited[nx][ny]:
                visited[nx][ny] = True
                stack.append((nx, ny))

    # Count the number of visited squares
    count = sum(sum(row) for row in visited)
    print(count)

solve()