from collections import deque

def min_moves_to_same_cell(grid):
    n = len(grid)
    player1_pos = None
    player2_pos = None
    
    # Find the positions of the two players
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                if player1_pos is None:
                    player1_pos = (i, j)
                else:
                    player2_pos = (i, j)
    
    # If the players are already in the same cell, return 0
    if player1_pos == player2_pos:
        return 0
    
    # Perform BFS to find the minimum number of moves
    queue = deque([(player1_pos, player2_pos, 0)])
    visited = set()
    
    while queue:
        p1, p2, moves = queue.popleft()
        if (p1, p2) in visited:
            continue
        visited.add((p1, p1))
        
        # Check all four directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_p1 = (p1[0] + dx, p1[1] + dy)
            new_p2 = (p2[0] + dx, p2[1] + dy)
            
            # Check if the new positions are valid and not obstacles
            if (0 <= new_p1[0] < n and 0 <= new_p1[1] < n and grid[new_p1[0]][new_p1[1]] != '#' and
                0 <= new_p2[0] < n and 0 <= new_p2[1] < n and grid[new_p2[0]][new_p2[1]] != '#'):
                if new_p1 == new_p2:
                    return moves + 1
                queue.append((new_p1, new_p2, moves + 1))
    
    return -1

# Read input from stdin
n = int(input())
grid = [input() for _ in range(n)]

# Solve the problem and print the answer
print(min_moves_to_same_cell(grid))