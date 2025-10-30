# YOUR CODE HERE
from collections import deque

def can_reach_goal(H, W, grid, medicines):
    # Find start and goal positions
    start = goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS queue: (current_row, current_col, current_energy)
    queue = deque([(start[0], start[1], 0)])
    visited = set()
    visited.add((start[0], start[1], 0))
    
    # Convert medicines list to a dictionary for quick access
    medicine_dict = {(r-1, c-1): e for r, c, e in medicines}
    
    while queue:
        r, c, energy = queue.popleft()
        
        # If we reach the goal, return True
        if (r, c) == goal:
            return True
        
        # Check if there's a medicine at the current cell
        if (r, c) in medicine_dict:
            new_energy = medicine_dict[(r, c)]
            if (r, c, new_energy) not in visited:
                visited.add((r, c, new_energy))
                queue.append((r, c, new_energy))
        
        # Try to move in all four directions
        if energy > 0:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                    if (nr, nc, energy - 1) not in visited:
                        visited.add((nr, nc, energy - 1))
                        queue.append((nr, nc, energy - 1))
    
    return False

# Read input
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]
N = int(input())
medicines = [tuple(map(int, input().split())) for _ in range(N)]

# Determine if Takahashi can reach the goal
if can_reach_goal(H, W, grid, medicines):
    print("Yes")
else:
    print("No")