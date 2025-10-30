import sys
from collections import deque

def min_moves_to_same_cell(grid):
    N = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find the positions of the two players
    player1_pos = None
    player2_pos = None
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                if player1_pos is None:
                    player1_pos = (i, j)
                else:
                    player2_pos = (i, j)
    
    # BFS to find the minimum number of moves to bring the two players to the same cell
    queue = deque([(player1_pos, player2_pos, 0)])
    visited = set()
    visited.add((player1_pos, player2_pos))
    
    while queue:
        (x1, y1), (x2, y2), moves = queue.popleft()
        
        # Check if the two players are in the same cell
        if (x1, y1) == (x2, y2):
            return moves
        
        # Try all four directions
        for dx, dy in directions:
            new_x1, new_y1 = x1 + dx, y1 + dy
            new_x2, new_y2 = x2 + dx, y2 + dy
            
            # Check if the new positions are within bounds and empty
            if 0 <= new_x1 < N and 0 <= new_y1 < N and grid[new_x1][new_y1] == '.':
                if 0 <= new_x2 < N and 0 <= new_y2 < N and grid[new_x2][new_y2] == '.':
                    if (new_x1, new_y1, new_x2, new_y2) not in visited:
                        visited.add((new_x1, new_y1, new_x2, new_y2))
                        queue.append(((new_x1, new_y1), (new_x2, new_y2), moves + 1))
    
    # If it is impossible to bring the two players to the same cell
    return -1

# Read input from stdin
N = int(sys.stdin.readline().strip())
grid = [sys.stdin.readline().strip() for _ in range(N)]

# Solve the problem and write the answer to stdout
result = min_moves_to_same_cell(grid)
print(result)