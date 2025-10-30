# YOUR CODE HERE
import sys

def solve():
    # Read the grid size N from standard input. N is guaranteed to be an odd integer.
    N = int(sys.stdin.readline())
    
    # Initialize an N x N grid with zeros. We will use 0-based indexing.
    grid = [[0] * N for _ in range(N)]
    
    # Calculate the coordinates of the center cell using 0-based indexing.
    # Since N is odd, N//2 gives the middle index.
    center_r, center_c = N // 2, N // 2

    # Set the starting position to the top-left corner (0, 0).
    r, c = 0, 0
    # Initialize the dragon part number to 1.
    k = 1
    
    # Define the changes in row and column for each direction:
    # 0: Right (0, +1)
    # 1: Down (+1, 0)
    # 2: Left (0, -1)
    # 3: Up (-1, 0)
    dr = [0, 1, 0, -1] 
    dc = [1, 0, -1, 0]
    # Start moving Right.
    dir_idx = 0
    
    # The initial length of path segments (Right, Down, Left) is N-1.
    current_len = N - 1
    
    # This counter tracks how many segments of the current length need to be completed 
    # before decreasing the length. Initially, it's 3 (for R, D, L segments of length N-1).
    # After this initial phase, the length decreases every 2 segments.
    change_len_after = 3 
    
    # This variable tracks the number of steps taken within the current segment.
    current_step_in_segment = 0
    
    # The total number of dragon parts to place is N*N - 1.
    # This corresponds to the number of steps the path takes.
    steps_total = N * N - 1

    # Loop to fill the grid with dragon parts 1 to N*N - 1 following a spiral path.
    # The loop runs exactly `steps_total` times.
    for _ in range(steps_total):
        # Place the current dragon part number 'k' into the grid at the current position (r, c).
        grid[r][c] = k
        # Increment the part number for the next cell.
        k += 1
        
        # Calculate the coordinates of the next cell based on the current direction.
        r_next = r + dr[dir_idx]
        c_next = c + dc[dir_idx]
        
        # Increment the count of steps taken in the current segment.
        current_step_in_segment += 1
        
        # Check if the current segment has been completed (reached `current_len` steps).
        if current_step_in_segment == current_len:
            # If completed, change direction to the next one in the cycle (R -> D -> L -> U -> R ...).
            dir_idx = (dir_idx + 1) % 4
            # Reset the step counter for the new segment.
            current_step_in_segment = 0
            
            # Decrement the counter that tracks when to change segment length.
            change_len_after -= 1
            # If this counter reaches zero, it's time to decrease the segment length.
            if change_len_after == 0:
                # Decrease the length for future segments.
                current_len -= 1
                # Reset the counter. From now on, length decreases every 2 segments.
                change_len_after = 2
        
        # Update the current position (r, c) to the next position for the next iteration.
        r, c = r_next, c_next

    # After the loop finishes, all N*N - 1 parts are placed.
    # The final position (r, c) calculated by the loop logic will be the center cell.
    # The problem requires Takahashi 'T' to be placed at the center.
    # Overwrite the cell value at the center coordinates with 'T'.
    grid[center_r][center_c] = 'T'

    # Print the resulting grid to standard output.
    for i in range(N):
        # For each row `i`, print its elements separated by spaces.
        # Convert each element to a string to handle both numbers and 'T' correctly.
        print(*(str(x) for x in grid[i]))

# Execute the solve function to run the program logic.
solve()