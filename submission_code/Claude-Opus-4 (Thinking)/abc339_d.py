from collections import deque

n = int(input())
grid = []
for _ in range(n):
    grid.append(input().strip())

# Find initial positions of the two players
players = []
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'P':
            players.append((i, j))

r1, c1 = players[0]
r2, c2 = players[1]

# BFS
queue = deque([(r1, c1, r2, c2, 0)])
visited = {(r1, c1, r2, c2)}

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

while queue:
    r1, c1, r2, c2, moves = queue.popleft()
    
    # Check if both players are at the same position
    if r1 == r2 and c1 == c2:
        print(moves)
        exit()
    
    # Try all four directions
    for dr, dc in directions:
        # Calculate new positions
        nr1, nc1 = r1 + dr, c1 + dc
        nr2, nc2 = r2 + dr, c2 + dc
        
        # Check if player 1 can move
        if not (0 <= nr1 < n and 0 <= nc1 < n and grid[nr1][nc1] != '#'):
            nr1, nc1 = r1, c1  # Player 1 doesn't move
        
        # Check if player 2 can move
        if not (0 <= nr2 < n and 0 <= nc2 < n and grid[nr2][nc2] != '#'):
            nr2, nc2 = r2, c2  # Player 2 doesn't move
        
        # Check if this state has been visited
        if (nr1, nc1, nr2, nc2) not in visited:
            visited.add((nr1, nc1, nr2, nc2))
            queue.append((nr1, nc1, nr2, nc2, moves + 1))

# If we reach here, it's impossible
print(-1)