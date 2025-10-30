from collections import deque

n = int(input())
grid = [input().strip() for _ in range(n)]

# Find the positions of the two players
positions = []
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'P':
            positions.append((i + 1, j + 1))  # Convert to 1-based index

pos1, pos2 = positions

visited = set()
queue = deque()
queue.append((pos1, pos2, 0))
visited.add((pos1, pos2))

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

found = False
while queue:
    a, b, steps = queue.popleft()
    if a == b:
        print(steps)
        found = True
        break
    for di, dj in directions:
        # Compute new position for a
        ni_a, nj_a = a[0] + di, a[1] + dj
        if 1 <= ni_a <= n and 1 <= nj_a <= n and grid[ni_a - 1][nj_a - 1] != '#':
            new_a = (ni_a, nj_a)
        else:
            new_a = a
        # Compute new position for b
        ni_b, nj_b = b[0] + di, b[1] + dj
        if 1 <= ni_b <= n and 1 <= nj_b <= n and grid[ni_b - 1][nj_b - 1] != '#':
            new_b = (ni_b, nj_b)
        else:
            new_b = b
        # Check if new state is visited
        state = (new_a, new_b)
        if state not in visited:
            visited.add(state)
            queue.append((new_a, new_b, steps + 1))

if not found:
    print(-1)