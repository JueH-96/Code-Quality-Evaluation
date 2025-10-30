# YOUR CODE HERE
import sys
from collections import deque

# Function to solve the problem
def solve():
    # Read grid dimensions N and M from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the grid configuration from standard input
    # The grid stores characters '#' for rock and '.' for ice.
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Define the starting position using 0-based indexing
    # The problem states the player starts at (2,2) in 1-based indexing,
    # which corresponds to (1,1) in 0-based indexing.
    start_pos = (1, 1)

    # Based on problem constraints, the starting square (2,2) is guaranteed to be ice ('.').
    # An explicit check could be added for robustness, but isn't necessary given the problem constraints.
    # Example:
    # if grid[start_pos[0]][start_pos[1]] == '#':
    #     print(0) # If start is rock, player cannot move, touches 0 squares (or 1 if start counts?)
    #              # The problem says start is ice, so this path is not taken.
    #     return

    # Initialize a deque (double-ended queue) for Breadth-First Search (BFS).
    # The queue stores coordinates (tuples) of squares where the player rests.
    # Start the BFS from the initial position.
    q = deque([start_pos])
    
    # Set to keep track of coordinates which have served as resting spots from which moves were explored.
    # This prevents processing the same resting spot multiple times, avoiding cycles and redundant work.
    visited_resting_spots = {start_pos}
    
    # Set to store all unique coordinates (squares) that the player has touched.
    # A square is touched if the player passes through it or rests on it.
    # Initialize with the starting square, as the player initially rests there.
    touched_squares = {start_pos}

    # Define the four possible directions of movement as (dr, dc) tuples:
    # Up: (-1, 0), Down: (1, 0), Left: (0, -1), Right: (0, 1)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

    # Perform BFS until the queue is empty (all reachable resting spots have been processed)
    while q:
        # Dequeue the next resting spot coordinate (r, c) to explore moves from
        curr_r, curr_c = q.popleft()

        # Try moving in each of the four directions from the current resting spot
        for dr, dc in directions:
            
            # Simulate the slide starting from the current resting spot (curr_r, curr_c)
            # path_r, path_c will track the player's position during the slide. Initially, it's the starting spot of the slide.
            path_r, path_c = curr_r, curr_c
            
            # Keep sliding in the chosen direction (dr, dc) as long as the next square is ice
            while True:
                # Calculate the coordinates of the next square in the direction
                next_r, next_c = path_r + dr, path_c + dc
                
                # Check the type of the next square.
                # The problem guarantees that the grid's outer periphery consists of rocks ('#').
                # This ensures the player will always hit a rock before moving outside the grid boundaries.
                # Therefore, explicit boundary checks (like `0 <= next_r < N`) are not needed.
                if grid[next_r][next_c] == '#':
                    # If the next square is a rock, the player stops moving before entering it.
                    # The current position (path_r, path_c) is the final resting spot for this move.
                    break
                
                # If the next square is ice ('.') , the player moves onto it.
                path_r, path_c = next_r, next_c
                
                # Add the coordinate of the square the player just moved onto to the set of touched squares.
                # Using a set automatically handles duplicate coordinates; each square is counted only once.
                touched_squares.add((path_r, path_c))

            # After the slide simulation loop finishes, (path_r, path_c) holds the coordinates of the final resting spot.
            final_pos = (path_r, path_c)

            # Check if this final resting spot has already been processed (i.e., added to `visited_resting_spots`).
            # A resting spot is processed only once to ensure termination and efficiency.
            if final_pos not in visited_resting_spots:
                # If it's a new resting spot that hasn't been processed yet:
                # Mark it as visited (processed) by adding it to the set.
                visited_resting_spots.add(final_pos)
                # Add it to the queue, so moves starting from this new spot will be explored in future iterations.
                q.append(final_pos)

    # After the BFS completes, all reachable resting spots have been found, and all squares
    # touched during any possible sequence of moves have been recorded in `touched_squares`.
    # The final answer is the total number of unique squares touched.
    print(len(touched_squares))

# Execute the solve function to run the program logic
solve()