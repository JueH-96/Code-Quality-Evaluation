# YOUR CODE HERE
import sys
from collections import deque

# The problem asks to count the number of connected components of sensors ('#')
# on a grid. Two sensors are considered connected if they are horizontally,
# vertically, or diagonally adjacent. This connectivity is transitive, meaning
# all sensors reachable from each other through a path of adjacent sensors
# form a single interacting group. This is a classic connected components
# problem on a grid graph with 8-connectivity.

# We can solve this using Breadth-First Search (BFS) or Depth-First Search (DFS).
# BFS is often preferred for grid traversal problems with potentially large
# connected components to avoid hitting Python's recursion depth limit.

def bfs(r, c, grid, visited, H, W):
    """
    Performs Breadth-First Search starting from cell (r, c) to find and mark
    all connected sensor cells in the current component.

    Args:
        r (int): Row index of the starting cell (0-indexed).
        c (int): Column index of the starting cell (0-indexed).
        grid (list[str]): The grid representing sensor locations, where '#' is a sensor and '.' is empty.
        visited (list[list[bool]]): 2D boolean array to keep track of visited cells.
        H (int): Number of rows in the grid.
        W (int): Number of columns in the grid.
    """
    # Use a deque for efficient queue operations (append and popleft)
    queue = deque([(r, c)])
    # Mark the starting cell as visited
    visited[r][c] = True

    # Define the 8 possible directions to move from a cell (dr, dc)
    # These represent horizontal, vertical, and diagonal neighbors.
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # Top-left, Top, Top-right
        ( 0, -1),          ( 0, 1),  # Left, Right
        ( 1, -1), ( 1, 0), ( 1, 1)   # Bottom-left, Bottom, Bottom-right
    ]

    # Process the queue until it's empty
    while queue:
        # Get the current cell (row, col) from the front of the queue
        curr_r, curr_c = queue.popleft()

        # Explore all 8 neighbors of the current cell
        for dr, dc in directions:
            # Calculate the coordinates of the neighbor cell
            nr = curr_r + dr
            nc = curr_c + dc

            # Check if the neighbor coordinates are within the grid boundaries
            if 0 <= nr < H and 0 <= nc < W:
                # Check if the neighbor cell contains a sensor ('#')
                # and if it has not been visited yet.
                # We only want to visit unvisited sensor cells to extend the current component.
                if grid[nr][nc] == '#' and not visited[nr][nc]:
                    # Mark the neighbor cell as visited
                    visited[nr][nc] = True
                    # Add the neighbor cell to the queue to explore its neighbors later
                    queue.append((nr, nc))

# Main execution block
if __name__ == "__main__":
    # Read the grid dimensions H and W from the first line of standard input
    # Using sys.stdin.readline is generally faster for competitive programming
    line = sys.stdin.readline().split()
    H = int(line[0])
    W = int(line[1])

    # Read the grid rows from the subsequent lines of standard input
    grid = []
    for _ in range(H):
        # Read each line and remove the trailing newline character using strip()
        grid.append(sys.stdin.readline().strip())

    # Initialize a 2D boolean array 'visited' of the same dimensions as the grid.
    # This helps us keep track of which sensor cells have already been
    # processed as part of a connected component.
    visited = [[False for _ in range(W)] for _ in range(H)]
    
    # Initialize the counter for the number of distinct connected sensor components.
    count = 0

    # Iterate through each cell in the grid
    for i in range(H):
        for j in range(W):
            # If the current cell contains a sensor ('#')
            # AND it has not been visited yet:
            # This indicates that we have found the starting point of a new,
            # undiscovered connected component of sensors.
            if grid[i][j] == '#' and not visited[i][j]:
                # Increment the component counter
                count += 1
                # Start a BFS traversal from this cell. This traversal will
                # find and mark all sensor cells belonging to this newly found component,
                # ensuring they are not counted again as part of a new component.
                bfs(i, j, grid, visited, H, W)

    # After iterating through the entire grid, the 'count' variable holds
    # the total number of distinct connected components of sensors.
    # Print the final count to standard output.
    print(count)