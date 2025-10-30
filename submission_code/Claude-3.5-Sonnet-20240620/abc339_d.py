# YOUR CODE HERE
from collections import deque

def bfs(grid, start1, start2):
    n = len(grid)
    visited = set()
    queue = deque([(start1, start2, 0)])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        (x1, y1), (x2, y2), moves = queue.popleft()

        if (x1, y1) == (x2, y2):
            return moves

        for dx, dy in directions:
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy

            if 0 <= nx1 < n and 0 <= ny1 < n and grid[nx1][ny1] != '#':
                new_pos1 = (nx1, ny1)
            else:
                new_pos1 = (x1, y1)

            if 0 <= nx2 < n and 0 <= ny2 < n and grid[nx2][ny2] != '#':
                new_pos2 = (nx2, ny2)
            else:
                new_pos2 = (x2, y2)

            new_state = (new_pos1, new_pos2)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_pos1, new_pos2, moves + 1))

    return -1

n = int(input())
grid = [input() for _ in range(n)]

players = []
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'P':
            players.append((i, j))

result = bfs(grid, players[0], players[1])
print(result)