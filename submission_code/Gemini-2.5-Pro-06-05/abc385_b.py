import sys

def main():
    """
    Reads the problem input, simulates Santa's movement, and prints the result.
    """
    # Read grid dimensions and initial position
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        H, W, X, Y = map(int, line1.split())
    except (IOError, ValueError):
        # Handle potential empty input or format errors gracefully
        return

    # Convert 1-based input coordinates to 0-based for list indexing
    r, c = X - 1, Y - 1

    # Read the grid layout
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Read the movement string
    T = sys.stdin.readline().strip()

    # Use a set to keep track of unique houses visited
    visited_houses = set()

    # A dictionary to map move characters to coordinate changes (delta_row, delta_col)
    move_deltas = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }

    # Simulate Santa's journey
    for move_char in T:
        # Determine the change in coordinates for the current move
        dr, dc = move_deltas[move_char]
        
        # Calculate the potential next position
        next_r, next_c = r + dr, c + dc

        # Check if the destination cell is passable.
        # The grid is guaranteed to be surrounded by walls ('#'),
        # so we don't need to check for out-of-bounds access.
        if grid[next_r][next_c] != '#':
            # If passable, update Santa's position
            r, c = next_r, next_c

            # Check if the new cell contains a house
            if grid[r][c] == '@':
                # Add the house's coordinates to the set of visited houses
                visited_houses.add((r, c))

    # Convert the final 0-based coordinates back to 1-based for the output
    final_X = r + 1
    final_Y = c + 1
    
    # The number of distinct visited houses is the size of the set
    house_count = len(visited_houses)

    # Print the final position and house count in the specified format
    print(final_X, final_Y, house_count)

if __name__ == "__main__":
    main()