# YOUR CODE HERE
from collections import deque

def min_moves_to_meet(N, grid):
    # Find the positions of the two players
    player_positions = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                player_positions.append((i, j))
    
    # Initial positions of player 1 and player 2
    (x1, y1), (x2, y2) = player_positions
    
    # Directions for movement: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS queue
    queue = deque([(x1, y1, x2, y2, 0)])  # (player1_x, player1_y, player2_x, player2_y, moves)
    visited = set([(x1, y1, x2, y2)])
    
    while queue:
        px1, py1, px2, py2, moves = queue.popleft()
        
        # Check if players are on the same cell
        if px1 == px2 and py1 == py2:
            return moves
        
        # Try all possible moves
        for dx, dy in directions:
            # Move player 1
            npx1, npy1 = px1 + dx, py1 + dy
            if not (0 <= npx1 < N and 0 <= npy1 < N) or grid[npx1][npy1] == '#':
                npx1, npy1 = px1, py1  # Player 1 can't move
            
            # Move player 2
            npx2, npy2 = px2 + dx, py2 + dy
            if not (0 <= npx2 < N and 0 <= npy2 < N) or grid[npx2][npy2] == '#':
                npx2, npy2 = px2, py2  # Player 2 can't move
            
            # New state
            new_state = (npx1, npy1, npx2, npy2)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((npx1, npy1, npx2, npy2, moves + 1))
    
    # If we exhaust the queue without finding a solution
    return -1

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
grid = data[1:N+1]

# Solve the problem
result = min_moves_to_meet(N, grid)
print(result)