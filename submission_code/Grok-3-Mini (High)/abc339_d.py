import sys
from collections import deque

def min_moves(N, grid, start1, start2):
    visited = set()
    r1_start, c1_start = start1
    r2_start, c2_start = start2
    queue = deque()
    queue.append((r1_start, c1_start, r2_start, c2_start, 0))
    visited.add((r1_start, c1_start, r2_start, c2_start))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    while queue:
        r1, c1, r2, c2, moves = queue.popleft()
        for dr, dc in directions:
            # New position for player 1
            if 0 <= r1 + dr < N and 0 <= c1 + dc < N and grid[r1 + dr][c1 + dc] != '#':
                new_r1 = r1 + dr
                new_c1 = c1 + dc
            else:
                new_r1 = r1
                new_c1 = c1
            # New position for player 2
            if 0 <= r2 + dr < N and 0 <= c2 + dc < N and grid[r2 + dr][c2 + dc] != '#':
                new_r2 = r2 + dr
                new_c2 = c2 + dc
            else:
                new_r2 = r2
                new_c2 = c2
            # Check if they meet
            if new_r1 == new_r2 and new_c1 == new_c2:
                return moves + 1
            # Check if new state not visited
            new_state = (new_r1, new_c1, new_r2, new_c2)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_r1, new_c1, new_r2, new_c2, moves + 1))
    return -1

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
grid = []
for _ in range(N):
    s = data[index]
    index += 1
    grid.append(s)

# Find start positions of players
start1 = None
start2 = None
for i in range(N):
    for j in range(N):
        if grid[i][j] == 'P':
            if start1 is None:
                start1 = (i, j)
            else:
                start2 = (i, j)

# Compute and print the answer
ans = min_moves(N, grid, start1, start2)
print(ans)