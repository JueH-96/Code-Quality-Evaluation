# YOUR CODE HERE
import sys
# Import deque for efficient queue operations in BFS
from collections import deque

# Function to solve the problem
def solve():
    # Read grid dimensions H (height/rows) and W (width/columns) from standard input
    H, W = map(int, sys.stdin.readline().split())
    # Read the grid state into a list of strings
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Create a boolean grid `is_mobile` of the same dimensions as the input grid.
    # is_mobile[r][c] will be True if the cell at (r, c) is empty ('.') 
    # and has no adjacent magnets ('#'). Otherwise, it will be False.
    is_mobile = [[False] * W for _ in range(H)]

    # Define relative coordinates for checking neighbors: Right, Left, Down, Up
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]

    # Precompute the mobility status for all cells in the grid
    for r in range(H):
        for c in range(W):
            # We only need to determine mobility for empty cells
            if S[r][c] == '.':
                is_safe = True # Assume the cell is safe (mobile) initially
                # Check all 4 adjacent cells (up, down, left, right)
                for i in range(4):
                    nr, nc = r + dx[i], c + dy[i]
                    # Check if the neighbor's coordinates are within the grid boundaries
                    if 0 <= nr < H and 0 <= nc < W:
                        # If an adjacent cell contains a magnet ('#')
                        if S[nr][nc] == '#':
                            is_safe = False # The current cell is not safe/mobile
                            break # No need to check other neighbors for this cell
                # If the cell remained safe after checking all its neighbors
                if is_safe:
                    # Mark this cell as mobile in our boolean grid
                    is_mobile[r][c] = True

    # Create a boolean grid `visited` to keep track of visited cells during BFS.
    # This prevents processing the same connected component multiple times.
    visited = [[False] * W for _ in range(H)]
    # Initialize `max_freedom` to 0. This will store the maximum degree of freedom found.
    # Note: The problem guarantees at least one empty cell, so the minimum possible max_freedom is 1.
    max_freedom = 0

    # Iterate through each cell of the grid
    for r in range(H):
        for c in range(W):
            # We are only interested in starting calculations from empty cells ('.')
            if S[r][c] == '.':
                
                # Case 1: The empty cell is not mobile.
                # Takahashi cannot move from such a cell.
                if not is_mobile[r][c]:
                    # The degree of freedom for a non-mobile empty cell is 1 (it can only reach itself).
                    # We update `max_freedom` if 1 is greater than the current maximum.
                    # This ensures that even if all empty cells are non-mobile (like Sample 2), 
                    # we correctly report 1 as the maximum freedom.
                    max_freedom = max(max_freedom, 1)
                
                # Case 2: The empty cell is mobile and has not been visited yet.
                # This indicates we've found the starting point of a new connected component
                # of mobile cells that hasn't been explored yet.
                elif not visited[r][c]:
                    
                    # Start a Breadth-First Search (BFS) to find all cells reachable from (r, c).
                    # The reachable set includes all cells in the connected mobile component
                    # plus all non-mobile empty cells directly adjacent to any cell in the component.
                    
                    # `component_nodes` will store the set of (row, col) tuples for cells 
                    # belonging to the current connected mobile component. Using a set ensures uniqueness.
                    component_nodes = set() 
                    # `adjacent_non_mobile` will store the set of (row, col) tuples for unique 
                    # non-mobile empty cells that are adjacent to any cell in the `component_nodes`.
                    adjacent_non_mobile = set()
                    
                    # Initialize the BFS queue with the starting mobile cell (r, c).
                    q = deque([(r, c)])
                    # Mark the starting cell as visited globally to avoid reprocessing it.
                    visited[r][c] = True
                    # Add the starting cell to the set of nodes in this component.
                    component_nodes.add((r, c))

                    # Perform BFS as long as the queue is not empty
                    while q:
                        # Dequeue the next cell to process
                        curr_r, curr_c = q.popleft()
                        
                        # Explore the 4 neighbors of the current cell
                        for i in range(4):
                            nr, nc = curr_r + dx[i], curr_c + dy[i]

                            # Check if the neighbor's coordinates are valid (within the grid boundaries)
                            if 0 <= nr < H and 0 <= nc < W:
                                # Subcase 2a: The neighbor is also a mobile cell
                                if is_mobile[nr][nc]:
                                    # If this mobile neighbor hasn't been visited yet in any previous BFS
                                    if not visited[nr][nc]:
                                        # Mark it as visited globally
                                        visited[nr][nc] = True
                                        # Add it to the BFS queue to explore its neighbors later
                                        q.append((nr, nc))
                                        # Add it to the set of nodes for the current component
                                        component_nodes.add((nr, nc))
                                # Subcase 2b: The neighbor is an empty cell ('.') but is not mobile
                                elif S[nr][nc] == '.': 
                                    # This non-mobile empty cell is reachable from the current component.
                                    # Add it to the `adjacent_non_mobile` set. 
                                    # The set automatically handles duplicates if multiple component cells are adjacent to it.
                                    adjacent_non_mobile.add((nr, nc))
                    
                    # After the BFS for this connected component is complete:
                    # Calculate the degree of freedom for any cell starting within this component.
                    # It's the total number of cells in the mobile component (`component_nodes`) 
                    # plus the total number of unique adjacent non-mobile empty cells (`adjacent_non_mobile`).
                    current_freedom = len(component_nodes) + len(adjacent_non_mobile)
                    # Update the overall maximum degree of freedom found so far across all components and non-mobile cells.
                    max_freedom = max(max_freedom, current_freedom)

    # After checking all cells, print the final maximum degree of freedom found.
    print(max_freedom)

# Standard boilerplate to ensure the `solve` function is called when the script is executed.
if __name__ == '__main__':
    solve()