from collections import deque

def min_moves_to_meet(N, grid):
    # Find the starting positions of the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))

    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # BFS initialization
    queue = deque([(players[0], players[1], 0)])
    visited = set([(players[0], players[1])])

    while queue:
        (x1, y1), (x2, y2), moves = queue.popleft()

        # If players meet
        if (x1, y1) == (x2, y2):
            return moves

        for dx, dy in directions:
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy

            # Check if the new positions are valid
            if 0 <= nx1 < N and 0 <= ny1 < N and grid[nx1][ny1] != '#':
                x1_new = nx1, ny1
            else:
                x1_new = (x1, y1)

            if 0 <= nx2 < N and 0 <= ny2 < N and grid[nx2][ny2] != '#':
                x2_new = nx2, ny2
            else:
                x2_new = (x2, y2)

            if (x1_new, x2_new) not in visited:
                visited.add((x1_new, x2_new))
                queue.append((x1_new[0], x1_new[1], x2_new[0], x2_new[1], moves + 1))

    return -1

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
grid = data[1:]

result = min_moves_to_meet(N, grid)
print(result)