from collections import deque

def solve():
    N, M = map(int, input().split())
    grid = []
    for _ in range(N):
        grid.append(input().strip())
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Set to keep track of visited ice squares
    visited = set()
    
    # Queue for BFS - stores positions we can rest at
    queue = deque()
    
    # Start position (2,2) -> (1,1) in 0-indexed
    start_row, start_col = 1, 1
    queue.append((start_row, start_col))
    visited.add((start_row, start_col))
    
    # BFS to find all reachable positions
    while queue:
        row, col = queue.popleft()
        
        # Try moving in each direction
        for dr, dc in directions:
            # Keep moving in this direction until we hit a rock
            curr_row, curr_col = row, col
            path_visited = set()  # Track squares visited in this move
            
            while True:
                next_row = curr_row + dr
                next_col = curr_col + dc
                
                # Check if next position is within bounds
                if 0 <= next_row < N and 0 <= next_col < M:
                    if grid[next_row][next_col] == '.':  # Ice square
                        curr_row, curr_col = next_row, next_col
                        path_visited.add((curr_row, curr_col))
                    else:  # Rock square
                        break
                else:
                    break
            
            # Add all squares visited in this path to our visited set
            for pos in path_visited:
                visited.add(pos)
            
            # If we moved at least one step, the final position is a new rest position
            if path_visited and (curr_row, curr_col) not in [pos for pos in queue]:
                # Check if this position hasn't been processed yet
                if (curr_row, curr_col) not in visited or len([pos for pos in queue if pos == (curr_row, curr_col)]) == 0:
                    # Add to queue only if we haven't processed this position as a starting point
                    queue.append((curr_row, curr_col))
    
    return len(visited)

# Better approach using a cleaner BFS
def solve_clean():
    N, M = map(int, input().split())
    grid = []
    for _ in range(N):
        grid.append(input().strip())
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    visited_squares = set()  # All ice squares we can touch
    visited_positions = set()  # Positions we can rest at (to avoid reprocessing)
    
    queue = deque()
    start_row, start_col = 1, 1  # (2,2) in 1-indexed = (1,1) in 0-indexed
    
    queue.append((start_row, start_col))
    visited_positions.add((start_row, start_col))
    visited_squares.add((start_row, start_col))
    
    while queue:
        row, col = queue.popleft()
        
        for dr, dc in directions:
            curr_row, curr_col = row, col
            
            # Move in this direction until hitting a rock
            while True:
                next_row = curr_row + dr
                next_col = curr_col + dc
                
                if (0 <= next_row < N and 0 <= next_col < M and 
                    grid[next_row][next_col] == '.'):
                    curr_row, curr_col = next_row, next_col
                    visited_squares.add((curr_row, curr_col))
                else:
                    break
            
            # If we moved and reached a new position we haven't processed
            if (curr_row, curr_col) != (row, col) and (curr_row, curr_col) not in visited_positions:
                queue.append((curr_row, curr_col))
                visited_positions.add((curr_row, curr_col))
    
    return len(visited_squares)

print(solve_clean())