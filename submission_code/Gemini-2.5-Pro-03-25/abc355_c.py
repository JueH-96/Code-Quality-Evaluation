# YOUR CODE HERE
import sys

def solve():
    # Read N (grid size) and T (number of turns) from the first line of input.
    # N is the dimension of the N x N grid.
    # T is the number of integers that will be announced.
    n, t = map(int, sys.stdin.readline().split())
    
    # Read the list of T integers announced over the turns.
    # a[i] is the integer announced on turn i+1.
    a = list(map(int, sys.stdin.readline().split()))

    # Precompute the (row, col) position for each number from 1 to N*N.
    # This allows O(1) lookup of a number's location in the grid.
    # We use a list `pos` where pos[k] stores the 0-based (row, col) tuple for number k.
    # The list size is N*N + 1. Index 0 is unused as numbers start from 1.
    # Initialize with a placeholder tuple, e.g., (0, 0).
    pos = [(0, 0)] * (n * n + 1) 
    for r in range(n):  # Iterate through rows (0-based)
        for c in range(n):  # Iterate through columns (0-based)
            # Calculate the number in the cell at (row r, col c) based on the problem's 1-based definition.
            # The number in cell (i, j) (1-based) is N * (i-1) + j.
            # Converting to 0-based (r, c): i = r + 1, j = c + 1.
            # Number = N * ((r + 1) - 1) + (c + 1) = N * r + c + 1.
            num = n * r + c + 1
            # Store the 0-based position (r, c) for this number `num`.
            pos[num] = (r, c)

    # Initialize counters to keep track of marked cells.
    # `row_counts[r]` stores the number of marked cells in row `r`.
    row_counts = [0] * n
    # `col_counts[c]` stores the number of marked cells in column `c`.
    col_counts = [0] * n
    # `diag1_count` stores the number of marked cells on the main diagonal (top-left to bottom-right).
    # This diagonal consists of cells where row index equals column index (r == c).
    diag1_count = 0
    # `diag2_count` stores the number of marked cells on the anti-diagonal (top-right to bottom-left).
    # This diagonal consists of cells where row index + column index equals N - 1 (r + c == n - 1).
    diag2_count = 0

    # Simulate the T turns. The loop variable `turn` represents the 0-based index of the turn.
    for turn in range(t):
        # Get the number announced in the current turn.
        # `a[turn]` is the number announced on turn `turn + 1`.
        num_announced = a[turn]
        
        # Find the 0-based (row, col) coordinates of the announced number using the precomputed `pos` list.
        # The problem constraints guarantee 1 <= num_announced <= N*N, so `pos[num_announced]` is valid.
        r, c = pos[num_announced]

        # Increment the count for the row `r` where the cell was marked.
        row_counts[r] += 1
        # Increment the count for the column `c` where the cell was marked.
        col_counts[c] += 1
        
        # Check if the marked cell lies on the main diagonal (r == c).
        is_diag1 = (r == c)
        if is_diag1:
            # If it's on the main diagonal, increment the main diagonal counter.
            diag1_count += 1
            
        # Check if the marked cell lies on the anti-diagonal (r + c == n - 1).
        is_diag2 = (r + c == n - 1)
        if is_diag2:
            # If it's on the anti-diagonal, increment the anti-diagonal counter.
            # Note: If N is odd, the center cell is on both diagonals. 
            # This logic correctly increments both `diag1_count` and `diag2_count` for the center cell.
            diag2_count += 1

        # Check if Bingo is achieved after marking the cell (r, c).
        # Bingo occurs if any of the following conditions are met:
        # 1. The row `r` now has N marked cells.
        # 2. The column `c` now has N marked cells.
        # 3. The cell is on the main diagonal (`is_diag1` is True) AND the main diagonal now has N marked cells (`diag1_count == n`).
        # 4. The cell is on the anti-diagonal (`is_diag2` is True) AND the anti-diagonal now has N marked cells (`diag2_count == n`).
        if row_counts[r] == n or \
           col_counts[c] == n or \
           (is_diag1 and diag1_count == n) or \
           (is_diag2 and diag2_count == n):
            
            # Bingo achieved! Print the current turn number (which is `turn + 1` since `turn` is 0-based).
            print(turn + 1)
            # Since we need the first turn Bingo is achieved, we exit the program immediately.
            return 

    # If the loop completes all T turns without achieving Bingo, print -1.
    print("-1")

# Call the main function `solve` to run the solution logic.
solve()