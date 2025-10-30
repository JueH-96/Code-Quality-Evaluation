# YOUR CODE HERE
import sys
# Using deque for efficient queue operations in BFS
from collections import deque

# Setting a higher recursion depth limit is generally good practice for competitive programming,
# although it's primarily relevant for recursive DFS, not iterative BFS.
# sys.setrecursionlimit(2000000) # Example limit, adjust as needed if using DFS

def solve():
    """
    Reads grid dimensions (H, W) and the grid layout from standard input.
    Counts the number of connected components of '#' symbols (sensors)
    using Breadth-First Search (BFS). Sensors are considered connected if
    they are adjacent horizontally, vertically, or diagonally (8-directional adjacency).
    Prints the total count of connected components to standard output.
    """

    # Read grid dimensions H (height/rows) and W (width/columns) from the first line.
    try:
        # Use map to convert space-separated strings to integers
        H, W = map(int, sys.stdin.readline().split())
    except ValueError:
        # Handle potential errors if the input line is not two integers
        print("Error: Invalid input format for H and W.", file=sys.stderr)
        sys.exit(1) # Exit indicating an error

    # Read the grid layout (H lines of W characters each)
    # .strip() removes leading/trailing whitespace, including the newline character
    # Store the grid as a list of strings
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # --- Optional Input Validation (Constraints usually guarantee correctness) ---
    # Basic checks can prevent errors with malformed test cases or local testing.
    # Check if H and W are within the expected range (1 to 1000)
    # if not (1 <= H <= 1000 and 1 <= W <= 1000):
    #     print(f"Constraints violated: H={H}, W={W}. Must be between 1 and 1000.", file=sys.stderr)
    #     sys.exit(1)
    # # Check if each row has the correct length W
    # for i in range(H):
    #     if len(S[i]) != W:
    #         print(f"Row {i} has incorrect length {len(S[i])}, expected {W}.", file=sys.stderr)
    #         sys.exit(1)
    #     # Check if each row contains only valid characters '#' or '.'
    #     if not all(c in '#.' for c in S[i]):
    #          print(f"Row {i} contains invalid characters. Only '#' and '.' allowed.", file=sys.stderr)
    #          sys.exit(1)
    # --- End Optional Validation ---

    # Initialize a 2D list (grid) to keep track of visited cells during BFS.
    # `visited[r][c]` is True if cell (r, c) has been visited, False otherwise.
    # Dimensions are H rows and W columns.
    visited = [[False for _ in range(W)] for _ in range(H)]

    # Initialize the counter for the number of connected components found.
    component_count = 0

    # Initialize a deque (double-ended queue) to manage cells for BFS.
    # Deques provide efficient O(1) append (enqueue) and popleft (dequeue) operations.
    queue = deque()

    # Define the 8 possible relative offsets (delta row, delta column)
    # for checking adjacent cells (horizontally, vertically, and diagonally).
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # Above row (left, middle, right)
        ( 0, -1),          ( 0, 1),  # Same row (left, right)
        ( 1, -1), ( 1, 0), ( 1, 1)   # Below row (left, middle, right)
    ]

    # Iterate through every cell (r, c) in the grid using nested loops.
    for r in range(H):
        for c in range(W):
            # Check if the current cell contains a sensor ('#') AND
            # has not already been visited as part of a previously found component.
            if S[r][c] == '#' and not visited[r][c]:
                # If both conditions are true, we've found the start of a new connected component.
                component_count += 1  # Increment the component counter.

                # Start a Breadth-First Search (BFS) from this cell (r, c) to find all
                # connected sensors in this component.
                queue.append((r, c))  # Add the starting cell to the BFS queue.
                visited[r][c] = True # Mark the starting cell as visited immediately.

                # Process the queue until it's empty (meaning all reachable cells
                # in the current component have been visited).
                while queue:
                    # Dequeue the next cell from the front of the queue to explore its neighbors.
                    current_r, current_c = queue.popleft()

                    # Explore all 8 neighbors of the current cell.
                    for dr, dc in directions:
                        # Calculate the coordinates of the potential neighbor.
                        next_r, next_c = current_r + dr, current_c + dc

                        # Check 1: Is the neighbor within the grid boundaries (0 <= row < H, 0 <= col < W)?
                        if 0 <= next_r < H and 0 <= next_c < W:
                            # Check 2: Is the neighbor cell a sensor ('#')?
                            # Check 3: Has this neighbor cell NOT been visited yet?
                            if S[next_r][next_c] == '#' and not visited[next_r][next_c]:
                                # If all conditions are met, it's a valid, unvisited sensor neighbor
                                # belonging to the current component.
                                visited[next_r][next_c] = True # Mark it as visited to avoid reprocessing.
                                queue.append((next_r, next_c)) # Add it to the queue to explore its neighbors later.

    # After iterating through all grid cells and performing BFS for each component found,
    # print the final count of connected components to standard output.
    print(component_count)

# Execute the main function to start the process when the script is run.
solve()