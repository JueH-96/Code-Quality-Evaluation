# YOUR CODE HERE
import sys

def solve():
    """
    Reads the grid, starting position, and move sequence,
    then simulates Takahashi's movement and prints the final position.
    """
    # Read grid dimensions
    try:
        H, W = map(int, sys.stdin.readline().split())

        # Read starting coordinates (1-indexed)
        Si, Sj = map(int, sys.stdin.readline().split())

        # Read the grid layout
        C = [sys.stdin.readline().strip() for _ in range(H)]

        # Read the move sequence
        X = sys.stdin.readline().strip()

    except (IOError, ValueError):
        # This handles potential empty lines or malformed input,
        # which can occur in some testing environments.
        return

    # Convert 1-indexed start coordinates to 0-indexed for internal use
    r, c = Si - 1, Sj - 1

    # Process each move in the sequence
    for move in X:
        # Calculate potential next coordinates based on the move command
        next_r, next_c = r, c
        if move == 'L':
            next_c -= 1
        elif move == 'R':
            next_c += 1
        elif move == 'U':
            next_r -= 1
        elif move == 'D':
            next_r += 1
        
        # Check if the move is valid. A move is valid if the target cell:
        # 1. is within the grid boundaries.
        # 2. is an empty cell ('.').
        # The 'and' operator uses short-circuit evaluation, which prevents
        # an IndexError by checking the bounds before accessing the grid.
        if 0 <= next_r < H and 0 <= next_c < W and C[next_r][next_c] == '.':
            # If the move is valid, update the current position.
            r, c = next_r, next_c
        # If the move is not valid, the position (r, c) remains unchanged,
        # as specified in the problem.

    # Convert the final 0-indexed coordinates back to 1-indexed and print them.
    print(r + 1, c + 1)

if __name__ == "__main__":
    solve()