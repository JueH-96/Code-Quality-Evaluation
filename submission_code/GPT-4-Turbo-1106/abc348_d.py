from collections import deque

def can_reach_goal(grid, H, W, medicines):
    # Find the start and goal positions
    start = goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    # Create a dictionary to store medicines' energy at their positions
    medicine_dict = {(r-1, c-1): e for r, c, e in medicines}
    
    # Initialize the queue with the start position and 0 energy
    queue = deque([(start[0], start[1], 0)])
    
    # Keep track of visited positions with a certain amount of energy
    visited = set()
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        x, y, energy = queue.popleft()
        
        # If we reach the goal, return True
        if (x, y) == goal:
            return True
        
        # If we have visited this position with equal or more energy, continue
        if (x, y, energy) in visited and visited[(x, y, energy)] >= energy:
            continue
        
        # Mark this position as visited with the current energy
        visited[(x, y, energy)] = energy
        
        # If there's a medicine at this position, use it
        if (x, y) in medicine_dict:
            energy = max(energy, medicine_dict[(x, y)])
        
        # Try moving in all directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if the new position is within the grid and not an obstacle
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and energy > 0:
                queue.append((nx, ny, energy - 1))
    
    # If we never reach the goal, return False
    return False

# Read input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]
N = int(input())
medicines = [tuple(map(int, input().split())) for _ in range(N)]

# Solve the problem and print the result
print("Yes" if can_reach_goal(grid, H, W, medicines) else "No")