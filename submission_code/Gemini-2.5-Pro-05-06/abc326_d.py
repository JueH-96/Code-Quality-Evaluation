import sys
import itertools

def solve_problem():
    N = int(sys.stdin.readline())
    R_str = sys.stdin.readline().strip()
    C_str = sys.stdin.readline().strip()

    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    char_to_idx = {'A': 0, 'B': 1, 'C': 2}
    
    # col_char_counts[j][char_idx] = count of char_idx in column j among rows 0..r-1
    col_char_counts = [[0] * 3 for _ in range(N)]

    # This list will store the solution grid if one is found.
    # It's defined in the outer scope and modified by the nested function.
    # `nonlocal` keyword is used for assignment in Python 3.
    # Here, we effectively use it like a global variable for the backtracking process.
    solution_grid_holder = [None] # Using a list to pass "by reference" effectively

    ALL_CHARS = ['A', 'B', 'C']
    
    # Pre-calculate combinations of 3 columns
    possible_col_sets = list(itertools.combinations(range(N), 3))

    def backtrack(r):
        if solution_grid_holder[0] is not None: # Already found a solution
             return True

        if r == N:
            # Base Case: All N rows have been filled.
            # Row conditions and R_str constraints are met by construction.
            # C_str constraints were checked when characters were first placed in columns.
            # Final check: Ensure each column has exactly one A, B, C.
            for j in range(N):
                if not (col_char_counts[j][char_to_idx['A']] == 1 and \
                        col_char_counts[j][char_to_idx['B']] == 1 and \
                        col_char_counts[j][char_to_idx['C']] == 1):
                    return False # A column does not have one of each A, B, C.
            
            # All conditions satisfied. Store the solution.
            solution_grid_holder[0] = [row_list[:] for row_list in grid] # Deep copy
            return True

        # Recursive step: Try to fill row 'r'
        char_for_R_constr = R_str[r] # Character that must be leftmost in this row
        other_two_chars = [ch for ch in ALL_CHARS if ch != char_for_R_constr]
        
        # Iterate over all possible sets of 3 columns for row r
        for cols_indices_tuple in possible_col_sets:
            # cols_indices_tuple is sorted, e.g., (c_left, c_mid, c_right)
            col_left = cols_indices_tuple[0]
            col_mid = cols_indices_tuple[1]
            col_right = cols_indices_tuple[2]

            # There are 2 permutations for placing other_two_chars in col_mid and col_right
            for p_idx in range(2):
                char_mid = other_two_chars[0] if p_idx == 0 else other_two_chars[1]
                char_right = other_two_chars[1] if p_idx == 0 else other_two_chars[0]
                
                # Tentative placement for row r:
                # char_for_R_constr in grid[r][col_left]
                # char_mid          in grid[r][col_mid]
                # char_right        in grid[r][col_right]
                current_row_placements = [
                    (col_left, char_for_R_constr),
                    (col_mid, char_mid),
                    (col_right, char_right)
                ]

                # Check validity of this set of placements for row r
                can_place_this_combination = True
                for col_j, char_val in current_row_placements:
                    char_val_idx = char_to_idx[char_val]
                    
                    # Check if char_val already exists in this column (in rows 0..r-1)
                    if col_char_counts[col_j][char_val_idx] > 0:
                        can_place_this_combination = False
                        break
                    
                    # Check C_str constraint if this char is the topmost in its column
                    # Column col_j is empty above row r if sum of its counts (A,B,C) is 0
                    is_col_empty_above_r = (sum(col_char_counts[col_j]) == 0)
                    if is_col_empty_above_r and char_val != C_str[col_j]:
                        can_place_this_combination = False
                        break
                
                if not can_place_this_combination:
                    continue # Try next permutation or next set of columns

                # If valid, make the placement and update column character counts
                for col_j, char_val in current_row_placements:
                    grid[r][col_j] = char_val
                    col_char_counts[col_j][char_to_idx[char_val]] += 1
                
                # Recurse for the next row
                if backtrack(r + 1):
                    return True # Solution found and propagated up

                # Backtrack: undo placement and column character counts
                for col_j, char_val in current_row_placements:
                    grid[r][col_j] = '.'
                    col_char_counts[col_j][char_to_idx[char_val]] -= 1
            
            # If a solution was found by a sub-branch (p_idx loop), no need to check here again.
            # The `if backtrack(r+1): return True` handles propagation.

        return False # No solution found for row 'r' with any choice

    # Initial call to the backtracking function for row 0
    if backtrack(0):
        print("Yes")
        # solution_grid_holder[0] now contains the solved grid
        for i in range(N):
            print("".join(solution_grid_holder[0][i]))
    else:
        print("No")

solve_problem()