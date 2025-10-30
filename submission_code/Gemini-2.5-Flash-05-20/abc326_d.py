import sys
from itertools import permutations

def solve():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Generate all possible permutations of column indices (0 to N-1).
    # These permutations will represent the column assignments for 'A', 'B', and 'C'
    # characters in each row.
    # For example, if P_A = (p0, p1, ..., pN-1), it means 'A' is placed at grid[0][p0],
    # grid[1][p1], ..., grid[N-1][pN-1].
    all_perms = list(permutations(range(N)))

    # Iterate through all combinations of three permutations (P_A, P_B, P_C).
    # P_A: maps row_idx -> col_idx for character 'A'
    # P_B: maps row_idx -> col_idx for character 'B'
    # P_C: maps row_idx -> col_idx for character 'C'
    for p_a in all_perms:
        for p_b in all_perms:
            for p_c in all_perms:
                # --- Condition Check 1: Distinct positions in each row ---
                # For each row i, the columns assigned to 'A', 'B', and 'C' must be distinct.
                is_valid_row_distinct = True
                for i in range(N):
                    if p_a[i] == p_b[i] or p_a[i] == p_c[i] or p_b[i] == p_c[i]:
                        is_valid_row_distinct = False
                        break
                if not is_valid_row_distinct:
                    continue # This combination is invalid, try the next one

                # --- Condition Check 2: Leftmost character in each row (R string) ---
                is_valid_leftmost = True
                for i in range(N): # For each row i
                    # Collect (column_index, character) for the three non-empty cells in this row
                    cells_in_row = []
                    cells_in_row.append((p_a[i], 'A'))
                    cells_in_row.append((p_b[i], 'B'))
                    cells_in_row.append((p_c[i], 'C'))
                    
                    # Sort by column index to find the leftmost character
                    cells_in_row.sort()
                    
                    if cells_in_row[0][1] != R[i]: # The character at the minimum column must match R[i]
                        is_valid_leftmost = False
                        break
                if not is_valid_leftmost:
                    continue # This combination is invalid, try the next one

                # --- Condition Check 3: Topmost character in each column (C string) ---
                # To efficiently check topmost characters, we need inverse mappings:
                # inv_P_X[col_idx] gives the row_idx where character X is placed in that column.
                inv_p_a = [0] * N
                inv_p_b = [0] * N
                inv_p_c = [0] * N
                for i in range(N):
                    inv_p_a[p_a[i]] = i
                    inv_p_b[p_b[i]] = i
                    inv_p_c[p_c[i]] = i

                is_valid_topmost = True
                for j in range(N): # For each column j
                    # Collect (row_index, character) for the three non-empty cells in this column
                    cells_in_col = []
                    cells_in_col.append((inv_p_a[j], 'A'))
                    cells_in_col.append((inv_p_b[j], 'B'))
                    cells_in_col.append((inv_p_c[j], 'C'))

                    # Sort by row index to find the topmost character
                    cells_in_col.sort()
                    
                    if cells_in_col[0][1] != C[j]: # The character at the minimum row must match C[j]
                        is_valid_topmost = False
                        break
                if not is_valid_topmost:
                    continue # This combination is invalid, try the next one
                
                # --- All conditions met! Construct and print the grid ---
                print("Yes")
                grid = [['.' for _ in range(N)] for _ in range(N)]
                for i in range(N):
                    grid[i][p_a[i]] = 'A'
                    grid[i][p_b[i]] = 'B'
                    grid[i][p_c[i]] = 'C'
                
                for row_arr in grid:
                    print("".join(row_arr))
                
                return # Exit the function after printing the first valid solution

    # If the loops complete without finding any valid combination
    print("No")

# Call the solve function to run the program
solve()