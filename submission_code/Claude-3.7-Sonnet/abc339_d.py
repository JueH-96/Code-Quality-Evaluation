from collections import deque

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    # Find the two players
    players = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                players.append((i, j))
    
    # Initial positions of the two players
    r1, c1 = players[0]
    r2, c2 = players[1]
    
    # BFS approach
    queue = deque([(r1, c1, r2, c2, 0)])  # (row1, col1, row2, col2, moves)
    visited = {(r1, c1, r2, c2)}
    
    # Four possible directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r1, c1, r2, c2, moves = queue.popleft()
        
        # Check if both players are at the same position
        if (r1, c1) == (r2, c2):
            return moves
        
        for dr, dc in directions:
            # Calculate new positions
            nr1, nc1 = r1 + dr, c1 + dc
            nr2, nc2 = r2 + dr, c2 + dc
            
            # Check if new positions are valid
            if not (0 <= nr1 < n and 0 <= nc1 < n and grid[nr1][nc1] != '#'):
                nr1, nc1 = r1, c1  # Stay in place if can't move
                
            if not (0 <= nr2 < n and 0 <= nc2 < n and grid[nr2][nc2] != '#'):
                nr2, nc2 = r2, c2  # Stay in place if can't move
            
            # If this state hasn't been visited, add it to the queue
            if (nr1, nc1, nr2, nc2) not in visited:
                visited.add((nr1, nc1, nr2, nc2))
                queue.append((nr1, nc1, nr2, nc2, moves + 1))
    
    return -1  # Return -1 if it's impossible to bring the players together

print(solve())