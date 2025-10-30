import sys

def solve():
    """
    Reads an odd integer N and generates one valid arrangement of Takahashi
    and the dragon parts on an N x N grid, printing the result to stdout.
    """
    # Read the size of the grid from standard input.
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle potential empty input or non-integer format.
        return

    # Initialize an N x N grid with a placeholder value (0) to mark empty cells.
    # This is a safe way to create a 2D list in Python.
    grid = [[0] * N for _ in range(N)]

    # Place Takahashi ('T') at the center. For an odd N, the 0-based index
    # of the center is (N // 2, N // 2).
    center = N // 2
    grid[center][center] = 'T'

    # --- Generate the spiral path for the dragon parts ---
    # We will trace a clockwise spiral starting from the top-left corner (0, 0).

    # (r, c) will be the current row and column for placing a number.
    r, c = 0, 0
    
    # The number to be placed, starting from 1.
    num = 1

    # Define the directions for a clockwise spiral: Right, Down, Left, Up.
    # Each tuple represents the change in (row, column) for one step.
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Start with the first direction (Right). dir_idx is the index into the directions list.
    dir_idx = 0

    # The loop runs N^2 - 1 times to place all dragon parts.
    for _ in range(N * N - 1):
        # Place the current number at the current position.
        grid[r][c] = num
        num += 1

        # Determine the potential next position by moving one step in the current direction.
        dr, dc = directions[dir_idx]
        next_r, next_c = r + dr, c + dc

        # Check if a turn is needed. A turn occurs if the next position is
        # either out of the grid bounds or already occupied (i.e., its value is not 0).
        # This condition also ensures the path avoids the central 'T'.
        if not (0 <= next_r < N and 0 <= next_c < N and grid[next_r][next_c] == 0):
            # Turn clockwise by advancing the direction index. The modulo operator (%)
            # ensures the index cycles through 0, 1, 2, 3.
            dir_idx = (dir_idx + 1) % 4
            # Get the new direction vector after turning.
            dr, dc = directions[dir_idx]

        # Update the current position for the next number.
        r += dr
        c += dc

    # Print the resulting grid in the specified format.
    for row in grid:
        # The `*` operator unpacks the list elements as arguments to print().
        # By default, print() separates arguments with a space.
        print(*row)

solve()