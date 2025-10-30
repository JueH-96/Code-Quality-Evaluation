# YOUR CODE HERE
import sys

# Setting a higher recursion depth might be necessary for larger K or complex grids,
# but given K <= 11, the default limit (usually 1000) should be sufficient.
# If RecursionError occurs uncomment the next line:
# sys.setrecursionlimit(2000) 

def solve():
    # Read input dimensions H (height), W (width) and path length K (number of moves)
    H, W, K = map(int, sys.stdin.readline().split())
    # Read the grid configuration. '.' represents an empty cell, '#' represents a blocked cell.
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Pre-calculate the total number of empty cells '.' in the grid.
    # Also, store the coordinates of these empty cells for efficient iteration later.
    num_empty_cells = 0
    empty_cells_coords = []
    for r in range(H):
        for c in range(W):
           if grid[r][c] == '.':
               num_empty_cells += 1
               empty_cells_coords.append((r, c)) # Store coordinates using 0-based indexing

    # A valid path consists of K+1 distinct empty cells.
    # If the total number of empty cells in the grid is less than K+1,
    # it's impossible to form such a path. Print 0 and exit early.
    if K + 1 > num_empty_cells:
        print(0)
        return

    # Define the Depth First Search (DFS) function to count valid paths.
    # Parameters:
    # r, c: current cell coordinates (0-based row and column).
    # moves_count: number of moves made so far to reach the current cell (starts at 0).
    # visited: a set containing coordinates (tuples) of cells visited in the current path.
    def dfs(r, c, moves_count, visited):
        # Base case: If K moves have been completed, we have found a valid path
        # consisting of K+1 cells. Return 1 to count this path.
        if moves_count == K:
            return 1

        # Initialize the count of paths extending from the current state (cell and path history).
        count = 0
        
        # Explore potential next moves to adjacent cells (up, down, left, right).
        # Define relative coordinate offsets for the four neighbors.
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        
        for dr, dc in neighbor_offsets:
            # Calculate the coordinates of the neighbor cell.
            nr, nc = r + dr, c + dc
            
            # Check if the neighbor cell (nr, nc) is a valid next step in the path:
            
            # 1. Check if the neighbor is within the grid boundaries.
            if 0 <= nr < H and 0 <= nc < W:
                # 2. Check if the neighbor cell is empty ('.').
                if grid[nr][nc] == '.':
                    # 3. Check if the neighbor cell has not already been visited in the current path.
                    # This ensures the path visits distinct cells.
                    if (nr, nc) not in visited:
                        
                        # If the neighbor cell is valid, proceed recursively:
                        # Add the neighbor to the 'visited' set before the recursive call.
                        # This marks it as visited for the path extension being explored.
                        visited.add((nr, nc))
                        
                        # Recursively call DFS starting from the neighbor cell.
                        # Increment the move count since one move has been made to reach the neighbor.
                        # Add the result (count of paths found from the neighbor) to the current count.
                        count += dfs(nr, nc, moves_count + 1, visited)
                        
                        # Backtrack: remove the neighbor from the 'visited' set after the recursive call returns.
                        # This step is crucial. It undoes the change made for this exploration branch,
                        # allowing other paths to be explored correctly. For example, paths that visit 
                        # (nr, nc) later, or paths from (r, c) that go to a different neighbor first.
                        visited.remove((nr, nc))
        
        # Return the total count of valid paths found that extend from the current cell (r, c)
        # given the current path history (visited set and moves_count).
        return count

    # Initialize the total count of valid paths found across all possible starting cells.
    total_paths = 0
    
    # Iterate through only the pre-calculated empty cells. Each empty cell can be a potential starting point.
    # This is slightly more efficient than iterating through all H*W cells, especially if the grid is sparse.
    for r_start, c_start in empty_cells_coords:
            # For each potential starting cell (r_start, c_start):
            
            # Initialize the 'visited' set for a new path starting at this cell.
            # The set initially contains only the starting cell itself.
            visited_init = set()
            visited_init.add((r_start, c_start))
            
            # Call the DFS function starting from this cell. The initial number of moves is 0.
            # The DFS will explore all valid paths of length K+1 starting from (r_start, c_start).
            # Add the count returned by DFS to the overall total path count.
            total_paths += dfs(r_start, c_start, 0, visited_init)

    # After checking all possible starting empty cells, print the final total count of valid paths.
    print(total_paths)

# Call the solve function to execute the main program logic.
solve()