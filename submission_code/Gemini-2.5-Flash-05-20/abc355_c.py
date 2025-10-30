import sys

def solve():
    # Read N (grid size) and T (number of turns)
    N, T = map(int, sys.stdin.readline().split())
    
    # Read the list of announced numbers A
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize counters for rows and columns.
    # row_counts[r_idx] stores the number of marked cells in the 0-indexed row r_idx.
    row_counts = [0] * N
    # col_counts[c_idx] stores the number of marked cells in the 0-indexed column c_idx.
    col_counts = [0] * N
    
    # Counters for the two diagonals.
    # main_diag_count for the top-left to bottom-right diagonal.
    main_diag_count = 0
    # anti_diag_count for the top-right to bottom-left diagonal.
    anti_diag_count = 0

    # Iterate through each turn from 0 to T-1 (representing turns 1 to T).
    for k in range(T):
        current_num = A[k]

        # Calculate the 0-indexed row (r_idx) and column (c_idx) for the current number.
        # The problem states that cell (i, j) (1-indexed) contains N*(i-1) + j.
        # Given a number 'val', its 0-indexed row is (val - 1) // N
        # and its 0-indexed column is (val - 1) % N.
        r_idx = (current_num - 1) // N
        c_idx = (current_num - 1) % N

        # Increment the count for the corresponding row and column.
        row_counts[r_idx] += 1
        col_counts[c_idx] += 1

        # Check if the cell is on the main diagonal (where 0-indexed row == 0-indexed column).
        if r_idx == c_idx:
            main_diag_count += 1
        
        # Check if the cell is on the anti-diagonal (where 0-indexed row + 0-indexed column == N - 1).
        # This translates from 1-indexed i + j == N + 1.
        if r_idx + c_idx == N - 1:
            anti_diag_count += 1
        
        # After marking the cell and updating counts, check if any Bingo condition is met.
        # Bingo is achieved if any row, any column, or any diagonal has N marked cells.
        if row_counts[r_idx] == N:
            print(k + 1) # Bingo achieved on turn k+1 (since k is 0-indexed).
            return
        
        if col_counts[c_idx] == N:
            print(k + 1)
            return
            
        if main_diag_count == N:
            print(k + 1)
            return
            
        if anti_diag_count == N:
            print(k + 1)
            return

    # If the loop completes without any Bingo being achieved, print -1.
    print(-1)

# Call the solve function to execute the program logic.
solve()