# YOUR CODE HERE
import sys

# Function to read input efficiently
def read_ints():
    """Reads a line of space-separated integers from stdin."""
    return map(int, sys.stdin.readline().split())

def read_str():
    """Reads a line string from stdin and removes leading/trailing whitespace."""
    return sys.stdin.readline().strip()

def solve():
    """Solves the problem based on the input specification."""
    
    # Read grid dimensions H (height/rows), W (width/columns) and number of moves N
    H, W, N = read_ints()
    
    # Read the sequence of moves T as a string
    T = read_str()
    
    # Read the grid layout S. S is a list of strings, where S[i] represents the i-th row (0-based index).
    # S[r][c] gives the character state ('.' for land, '#' for sea) of the cell at row r, column c (0-based indices).
    S = [read_str() for _ in range(H)]

    # Use a set to store distinct possible final positions.
    # Positions are stored as tuples (row, column) using 0-based indexing.
    # Using a set automatically handles uniqueness: if a final position is reached via multiple valid paths,
    # it's only stored once in the set.
    possible_final_positions = set()

    # Iterate over all cells in the grid to check if they can be a valid starting position.
    # A valid starting cell must satisfy two conditions:
    # 1. It must be a land cell ('.').
    # 2. It must not be on the perimeter of the grid.
    # The perimeter consists of rows 0 and H-1, and columns 0 and W-1.
    # Therefore, potential starting cells must have row index `r0` in the range [1, H-2]
    # and column index `c0` in the range [1, W-2].
    for r0 in range(1, H - 1):
        for c0 in range(1, W - 1):
            
            # Check condition 1: If the cell (r0, c0) is a sea cell ('#'), it cannot be a starting point. Skip it.
            if S[r0][c0] == '#':
                continue

            # If (r0, c0) is a land cell and not on the perimeter, it's a potential starting point.
            # Simulate Takahashi's path starting from this cell.
            curr_r, curr_c = r0, c0  # Initialize current position to the starting cell
            valid_path = True  # Flag to track if the path stays entirely on land cells

            # Perform N moves according to the instruction string T
            for move in T:
                # Update current coordinates based on the move instruction
                if move == 'L':
                    curr_c -= 1  # Move left
                elif move == 'R':
                    curr_c += 1  # Move right
                elif move == 'U':
                    curr_r -= 1  # Move up
                elif move == 'D':
                    curr_r += 1  # Move down
                
                # Check if the new cell (curr_r, curr_c) is a sea cell ('#').
                # Access the grid state using S[curr_r][curr_c].
                # The problem guarantees that all cells on the perimeter are sea ('#').
                # Therefore, if Takahashi moves onto a perimeter cell, S[curr_r][curr_c] will be '#'.
                # This check correctly identifies invalid moves onto sea or perimeter cells.
                if S[curr_r][curr_c] == '#':
                    valid_path = False  # Mark the path starting from (r0, c0) as invalid
                    break  # Stop simulating moves for this starting cell, as it hit sea.

            # After the loop finishes (either after N moves or breaking early):
            # Check if the path remained valid throughout the entire sequence of N moves.
            if valid_path:
                # If the path was valid, the final position (curr_r, curr_c) is a possible outcome.
                # Add this final position to the set. Duplicates are automatically handled by the set data structure.
                possible_final_positions.add((curr_r, curr_c))

    # The answer to the problem is the total number of distinct final positions found.
    # This is equal to the number of unique elements in the `possible_final_positions` set.
    print(len(possible_final_positions))

# Standard execution block for Python scripts: ensures `solve()` is called only when the script is run directly.
if __name__ == '__main__':
    solve()