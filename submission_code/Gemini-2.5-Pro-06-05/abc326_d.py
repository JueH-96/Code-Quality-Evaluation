import sys

def solve():
    """
    Reads input, solves the grid puzzle using backtracking, and prints the result.
    """
    try:
        # Read problem inputs
        N = int(sys.stdin.readline())
        R = sys.stdin.readline().strip()
        C = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handles cases like empty input at the end of a file stream.
        return

    # p_a[i] will store the column where 'A' is placed in row i.
    # Similarly for p_b and p_c. These arrays together define the grid.
    p_a = [-1] * N
    p_b = [-1] * N
    p_c = [-1] * N

    # ans_grid is a mutable container (a list) to hold the final grid.
    # This allows the nested backtrack function to modify it from any recursion depth.
    ans_grid = [None]

    def backtrack(row_idx, used_a, used_b, used_c):
        """
        Recursively tries to build a valid grid row by row.

        Args:
            row_idx: The current row index to be filled (from 0 to N-1).
            used_a: A set of column indices already occupied by 'A's in previous rows.
            used_b: A set of column indices already occupied by 'B's in previous rows.
            used_c: A set of column indices already occupied by 'C's in previous rows.
        """
        # If a solution has been found in another recursive branch, stop this one.
        if ans_grid[0] is not None:
            return

        # Base case: If all rows have been successfully assigned characters.
        if row_idx == N:
            # A potential solution is found based on row constraints.
            # Now, we must verify the column constraints.
            
            # For faster lookup, create inverse mappings: inv_p_a[j] = row of 'A' in column j
            inv_pa = [-1] * N
            inv_pb = [-1] * N
            inv_pc = [-1] * N
            for i in range(N):
                inv_pa[p_a[i]] = i
                inv_pb[p_b[i]] = i
                inv_pc[p_c[i]] = i

            for j in range(N):
                row_a, row_b, row_c = inv_pa[j], inv_pb[j], inv_pc[j]
                
                # Determine the topmost character in column j by finding the minimum row index.
                topmost_char = ''
                if row_a < row_b and row_a < row_c:
                    topmost_char = 'A'
                elif row_b < row_a and row_b < row_c:
                    topmost_char = 'B'
                else:  # row_c must be the smallest
                    topmost_char = 'C'

                # Check if the topmost character matches the column requirement.
                if topmost_char != C[j]:
                    return  # This candidate solution is invalid.

            # If all column constraints are met, we have found a valid solution.
            # Construct the grid and store it.
            grid = [['.'] * N for _ in range(N)]
            for i in range(N):
                grid[i][p_a[i]] = 'A'
                grid[i][p_b[i]] = 'B'
                grid[i][p_c[i]] = 'C'
            ans_grid[0] = grid
            return

        # --- Recursive Step ---
        # Try to fill the current row `row_idx`.
        
        # Get lists of columns that are not yet used for each character.
        avail_a = [j for j in range(N) if j not in used_a]
        avail_b = [j for j in range(N) if j not in used_b]
        avail_c = [j for j in range(N) if j not in used_c]

        # Iterate through all possible combinations of column choices for A, B, and C.
        for ca in avail_a:
            for cb in avail_b:
                for cc in avail_c:
                    # The three chosen columns must be distinct.
                    if len({ca, cb, cc}) < 3:
                        continue
                    
                    # Check the row-start constraint for the current row.
                    leftmost_col = min(ca, cb, cc)
                    leftmost_char_placed = ''
                    if leftmost_col == ca:
                        leftmost_char_placed = 'A'
                    elif leftmost_col == cb:
                        leftmost_char_placed = 'B'
                    else:  # leftmost_col == cc
                        leftmost_char_placed = 'C'
                    
                    if leftmost_char_placed != R[row_idx]:
                        continue # This combination violates the row-start constraint.

                    # If the choice is valid for this row, assign it and recurse.
                    p_a[row_idx] = ca
                    p_b[row_idx] = cb
                    p_c[row_idx] = cc
                    
                    # Proceed to the next row with updated sets of used columns.
                    # The `|` operator creates a new set for the recursive call.
                    backtrack(row_idx + 1, used_a | {ca}, used_b | {cb}, used_c | {cc})
                    
                    # If the recursive call found a solution, we can stop searching.
                    if ans_grid[0] is not None:
                        return
    
    # Start the backtracking search from the first row (index 0) with empty sets.
    backtrack(0, set(), set(), set())

    # After the search is complete, print the result.
    if ans_grid[0]:
        print("Yes")
        for row in ans_grid[0]:
            print("".join(row))
    else:
        print("No")

solve()