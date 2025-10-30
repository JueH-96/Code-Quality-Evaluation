# YOUR CODE HERE
def solve(N, M, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()
    queue = [(2, 2)]
    visited.add((2, 2))
    
    while queue:
        x, y = queue.pop(0)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 1 <= nx < N and 1 <= ny < M and grid[nx][ny] == '.':
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                nx += dx
                ny += dy
    
    return len(visited)

import sys
input = sys.stdin.read
data = input().split()

N, M = map(int, data[0].split())
grid = [list(row) for row in data[1:]]
print(solve(N, M, grid))