from collections import deque

def solve():
    n = int(input())
    grid = []
    players = []
    
    for i in range(n):
        row = input().strip()
        grid.append(row)
        for j in range(n):
            if row[j] == 'P':
                players.append((i, j))
    
    # Starting positions
    start1, start2 = players[0], players[1]
    
    # BFS
    queue = deque([(start1[0], start1[1], start2[0], start2[1], 0)])
    visited = set()
    visited.add((start1[0], start1[1], start2[0], start2[1]))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r1, c1, r2, c2, moves = queue.popleft()
        
        # Check if players are at the same position
        if r1 == r2 and c1 == c2:
            return moves
        
        # Try all four directions
        for dr, dc in directions:
            # Calculate new positions
            nr1, nc1 = r1 + dr, c1 + dc
            nr2, nc2 = r2 + dr, c2 + dc
            
            # Check if player 1 can move
            if (0 <= nr1 < n and 0 <= nc1 < n and grid[nr1][nc1] != '#'):
                # Player 1 moves
                pass
            else:
                # Player 1 doesn't move
                nr1, nc1 = r1, c1
            
            # Check if player 2 can move
            if (0 <= nr2 < n and 0 <= nc2 < n and grid[nr2][nc2] != '#'):
                # Player 2 moves
                pass
            else:
                # Player 2 doesn't move
                nr2, nc2 = r2, c2
            
            # Check if this state has been visited
            state = (nr1, nc1, nr2, nc2)
            if state not in visited:
                visited.add(state)
                queue.append((nr1, nc1, nr2, nc2, moves + 1))
    
    return -1

print(solve())