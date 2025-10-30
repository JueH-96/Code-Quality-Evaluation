import sys

def solve():
    """
    Solves the Bingo problem by reading grid dimensions and announced numbers,
    then determining the first turn a bingo is achieved.
    """
    
    # Read problem inputs: N (grid size) and T (number of turns).
    # It's good practice to assume input is well-formed as per problem constraints.
    try:
        N, T = map(int, sys.stdin.readline().split())
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # This handles potential empty lines or malformed input,
        # which is unlikely in a competitive programming environment
        # but is robust.
        return

    # Initialize counters for rows, columns, and the two main diagonals.
    # We use 0-indexed arrays for simplicity.
    row_counts = [0] * N
    col_counts = [0] * N
    diag1_count = 0  # Counter for the main diagonal (top-left to bottom-right)
    diag2_count = 0  # Counter for the anti-diagonal (top-right to bottom-left)

    # Process each announced number turn by turn.
    for turn, num in enumerate(A, 1):
        # Calculate the 0-indexed row (r) and column (c) for the announced number.
        # The problem states the value in an 1-indexed cell (i, j) is N*(i-1) + j.
        # For a 0-indexed cell (r, c), the value is N*r + c + 1.
        # So, num = N*r + c + 1, which means num - 1 = N*r + c.
        # From this, we can find r and c using integer division and modulo.
        r = (num - 1) // N
        c = (num - 1) % N

        # Mark the cell by incrementing the relevant counters.
        row_counts[r] += 1
        col_counts[c] += 1

        # Check if the cell lies on a diagonal and update the diagonal counters.
        # A cell is on the main diagonal if its row and column indices are equal.
        is_on_diag1 = (r == c)
        # A cell is on the anti-diagonal if r + c = N - 1 for a 0-indexed grid.
        is_on_diag2 = (r + c == N - 1)

        if is_on_diag1:
            diag1_count += 1
        
        # A cell can be on both diagonals (the center of an odd-sized grid).
        # This check is separate from the one above to handle this case correctly.
        if is_on_diag2:
            diag2_count += 1

        # Check if a bingo has been achieved on this turn.
        # A bingo occurs if a row, column, or diagonal that the current number is on
        # becomes full. We only need to check the lines affected by the current number.
        if (row_counts[r] == N or
            col_counts[c] == N or
            (is_on_diag1 and diag1_count == N) or
            (is_on_diag2 and diag2_count == N)):
            
            # If a bingo is achieved, print the current turn number and terminate the program.
            print(turn)
            return

    # If the loop completes, it means no bingo was achieved within T turns.
    # In this case, print -1 as required.
    print(-1)

# Run the solution
solve()