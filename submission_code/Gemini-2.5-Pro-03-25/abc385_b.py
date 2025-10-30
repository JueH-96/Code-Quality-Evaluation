# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read grid dimensions H, W and starting position X, Y (1-based)
    H, W, X, Y = map(int, sys.stdin.readline().split())
    
    # Convert starting position X, Y to 0-based indices for internal use
    # Problem statement uses 1-based indexing (rows 1..H, cols 1..W)
    # Python uses 0-based indexing (rows 0..H-1, cols 0..W-1)
    curr_r = X - 1
    curr_c = Y - 1
    
    # Read the grid layout
    grid = []
    for _ in range(H):
        # Read each row as a string and remove trailing newline/whitespace
        grid.append(sys.stdin.readline().strip())
    
    # Read the movement command string T
    T = sys.stdin.readline().strip()
    
    # Initialize a set to keep track of unique coordinates of visited houses
    # The coordinates stored will be 0-based indices (row, col)
    visited_houses = set()
    
    # The problem guarantees the starting cell S_{X,Y} is '.', which means it's passable and empty.
    # Therefore, Santa does not start in a house, and we don't need to check the initial cell.
    
    # Simulate Santa's movements according to the command string T
    for move in T:
        # Determine the potential next position based on the current move command
        next_r, next_c = curr_r, curr_c
        
        if move == 'U':
            next_r -= 1  # Move Up means decreasing row index
        elif move == 'D':
            next_r += 1  # Move Down means increasing row index
        elif move == 'L':
            next_c -= 1  # Move Left means decreasing column index
        elif move == 'R':
            next_c += 1  # Move Right means increasing column index
        
        # Check if the potential next cell is within the grid boundaries
        # Check 1: Row index is valid (0 <= next_r < H)
        # Check 2: Column index is valid (0 <= next_c < W)
        # Check 3: The cell is passable (not an impassable wall '#')
        if 0 <= next_r < H and 0 <= next_c < W and grid[next_r][next_c] != '#':
            # If the move is valid (within bounds and into a passable cell '.' or '@'):
            # Update Santa's current position to the new cell
            curr_r = next_r
            curr_c = next_c
            
            # Check if the cell Santa just moved into contains a house ('@')
            if grid[curr_r][curr_c] == '@':
                # If it's a house, add its 0-based coordinates to the set of visited houses.
                # Using a set ensures that each distinct house location is counted only once,
                # even if Santa visits the same house multiple times.
                visited_houses.add((curr_r, curr_c))
        # Else (if the move is invalid because it's out of bounds or into a wall '#'):
            # Santa stays in his current cell. His position (curr_r, curr_c) remains unchanged.
            # No need to check for houses again as he hasn't moved to a new cell.
            pass # This 'pass' is just for clarity, Python does nothing here anyway.

    # After processing all movements in T:
    # Convert the final 0-based position (curr_r, curr_c) back to 1-based for output
    final_X = curr_r + 1
    final_Y = curr_c + 1
    
    # The number of distinct houses visited is the size of the set `visited_houses`
    count_C = len(visited_houses)
    
    # Print the final position (1-based) and the count of distinct houses visited,
    # separated by spaces, as required by the problem statement.
    print(f"{final_X} {final_Y} {count_C}")

# Call the solve function to execute the logic when the script is run
solve()