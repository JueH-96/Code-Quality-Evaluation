import sys

def solve():
    """
    Reads a grid of characters, finds a unique 5-cell sequence spelling "snuke",
    and prints the 1-based coordinates of these cells.
    """
    # Read grid dimensions H (height/rows) and W (width/columns)
    try:
        H, W = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle potential empty input or malformed line
        return

    # Read the grid into a list of strings
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # The target word to find
    target = "snuke"
    target_length = len(target)

    # Define the 8 directions for searching (vertical, horizontal, and diagonal).
    # (dr, dc) represents the change in row and column for each step.
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
        (0, -1),           (0, 1),   # Left, Right
        (1, -1),  (1, 0),  (1, 1)    # Down-left, Down, Down-right
    ]

    # Iterate through every cell (start_r, start_c) of the grid.
    # Each cell is a potential starting point for the "snuke" sequence.
    for start_r in range(H):
        for start_c in range(W):
            # For each potential start, check all 8 possible directions.
            for dr, dc in directions:
                
                # A list to store the coordinates of a potential "snuke" sequence.
                path_coords = []
                
                # Assume we have a match until proven otherwise.
                is_match = True

                # Check for the 5 characters of "snuke" along the current direction.
                for i in range(target_length):
                    # Calculate the coordinates for the i-th character of the word.
                    curr_r = start_r + i * dr
                    curr_c = start_c + i * dc

                    # 1. Boundary Check: The current coordinates must be inside the grid.
                    if not (0 <= curr_r < H and 0 <= curr_c < W):
                        is_match = False
                        break  # Path goes out of bounds.

                    # 2. Character Check: The character at the grid cell must match the target.
                    if grid[curr_r][curr_c] != target[i]:
                        is_match = False
                        break  # Character mismatch.

                    # If checks pass, store the 1-based coordinates.
                    # The problem asks for 1-based indexing in the output.
                    path_coords.append((curr_r + 1, curr_c + 1))

                # If the loop completed without breaking, is_match is still True,
                # meaning we found the entire "snuke" word.
                if is_match:
                    # The problem guarantees a unique solution.
                    # Print the found coordinates and terminate the program.
                    for r, c in path_coords:
                        print(r, c)
                    # Exit successfully after finding the solution.
                    sys.exit(0)

# Run the solution
solve()