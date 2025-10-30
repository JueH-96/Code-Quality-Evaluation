from collections import deque

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# The pattern we need to follow
pattern = "snuke"

# BFS to find if there's a valid path
def can_reach():
    # Check if starting cell has 's'
    if grid[0][0] != 's':
        return False
    
    # Queue stores (row, col, position_in_pattern)
    queue = deque([(0, 0, 0)])
    # Visited set stores (row, col, position_in_pattern) to handle cycles
    visited = set([(0, 0, 0)])
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        row, col, pos = queue.popleft()
        
        # Check if we reached the destination
        if row == H - 1 and col == W - 1:
            return True
        
        # Try all four directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < H and 0 <= new_col < W:
                # Next position in pattern
                next_pos = (pos + 1) % 5
                
                # Check if the cell has the correct letter
                if grid[new_row][new_col] == pattern[next_pos]:
                    state = (new_row, new_col, next_pos)
                    if state not in visited:
                        visited.add(state)
                        queue.append(state)
    
    return False

if can_reach():
    print("Yes")
else:
    print("No")