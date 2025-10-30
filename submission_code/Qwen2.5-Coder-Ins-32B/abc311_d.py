# YOUR CODE HERE
def can_reach(grid, n, m):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()
    stack = [(2, 2)]
    visited.add((2, 2))
    
    while stack:
        x, y = stack.pop()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 1 <= nx <= n and 1 <= ny <= m and grid[nx][ny] == '.':
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                nx += dx
                ny += dy
    
    return len(visited)

import sys
input = sys.stdin.read().split()
n, m = map(int, input[:2])
grid = ['.' * (m + 2)] + ['.' + s + '.' for s in input[2:]] + ['.' * (m + 2)]

print(can_reach(grid, n + 1, m + 1))