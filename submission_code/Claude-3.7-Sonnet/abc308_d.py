from collections import deque

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

# Define the snuke pattern
snuke = "snuke"

# Define the BFS
def bfs():
    # Check if the starting position has the first letter of snuke
    if grid[0][0] != snuke[0]:
        return False
    
    visited = set([(0, 0, 0)])  # (row, col, pattern_index)
    queue = deque([(0, 0, 0)])
    
    while queue:
        row, col, pattern_index = queue.popleft()
        
        # Check if we've reached the destination
        if row == H - 1 and col == W - 1:
            return True
        
        # Define adjacent cells
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if valid move
            if 0 <= new_row < H and 0 <= new_col < W:
                next_pattern_index = (pattern_index + 1) % 5
                
                # Check if the letter matches the expected letter in the snuke pattern
                if grid[new_row][new_col] == snuke[next_pattern_index]:
                    if (new_row, new_col, next_pattern_index) not in visited:
                        visited.add((new_row, new_col, next_pattern_index))
                        queue.append((new_row, new_col, next_pattern_index))
    
    return False

if bfs():
    print("Yes")
else:
    print("No")