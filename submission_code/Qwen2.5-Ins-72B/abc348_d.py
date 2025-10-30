from collections import deque

def can_reach_goal(H, W, grid, N, medicines):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = None
    goal = None
    
    # Find start and goal positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    # Convert medicines to a dictionary for quick lookup
    medicine_dict = {(r-1, c-1): e for r, c, e in medicines}
    
    # BFS to find the shortest path to the goal
    queue = deque([(start[0], start[1], 0)])
    visited = set([(start[0], start[1], 0)])
    
    while queue:
        x, y, energy = queue.popleft()
        
        # Check if we have reached the goal
        if (x, y) == goal:
            return "Yes"
        
        # Check if we can use a medicine
        if (x, y) in medicine_dict:
            new_energy = medicine_dict[(x, y)]
            if (x, y, new_energy) not in visited:
                visited.add((x, y, new_energy))
                queue.append((x, y, new_energy))
        
        # Explore all possible moves
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and energy > 0:
                if (nx, ny, energy - 1) not in visited:
                    visited.add((nx, ny, energy - 1))
                    queue.append((nx, ny, energy - 1))
    
    return "No"

# Read input
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]
N = int(input())
medicines = [tuple(map(int, input().split())) for _ in range(N)]

# Solve and print the result
print(can_reach_goal(H, W, grid, N, medicines))