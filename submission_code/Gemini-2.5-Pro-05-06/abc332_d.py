import sys
import itertools

# Helper function to count inversions in a permutation
def count_inversions(p_tuple):
    # p_tuple is a tuple (or list) representing a permutation, e.g., (2, 0, 1)
    # This function counts pairs (i, j) such that i < j and p_tuple[i] > p_tuple[j]
    n = len(p_tuple)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p_tuple[i] > p_tuple[j]:
                inv_count += 1
    return inv_count

# Main logic
def solve():
    H, W = map(int, sys.stdin.readline().split())

    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))

    B = []
    for _ in range(H):
        B.append(list(map(int, sys.stdin.readline().split())))

    min_ops = float('inf')

    # Original 0-based indices for rows and columns
    row_indices_orig = list(range(H))
    col_indices_orig = list(range(W))

    # Iterate over all permutations of row indices
    # P_r is a tuple, e.g., for H=3, P_r could be (0, 1, 2), (0, 2, 1), etc.
    # P_r[i] means that the i-th row of the permuted grid will be row P_r[i] from the original grid A.
    for P_r in itertools.permutations(row_indices_orig):
        inv_r = count_inversions(P_r)

        # Pruning optimization: if row_inversions alone >= current min_ops, skip
        if inv_r >= min_ops:
            continue

        # Iterate over all permutations of column indices
        # P_c works similarly for columns.
        # P_c[j] means that the j-th col of the permuted grid will be col P_c[j] from the original grid A (after rows permuted).
        for P_c in itertools.permutations(col_indices_orig):
            inv_c = count_inversions(P_c)
            
            current_total_ops = inv_r + inv_c
            
            # Pruning optimization: if current_total_ops >= current min_ops, skip
            if current_total_ops >= min_ops:
                continue

            # Check if grid A, after applying P_r and P_c, matches grid B
            matches_B = True
            for r_idx_target in range(H):  # Iterate through rows of target grid B
                for c_idx_target in range(W):  # Iterate through columns of target grid B
                    # The element in A permuted by P_r and P_c at [r_idx_target][c_idx_target]
                    # is A[original_row_P_r[r_idx_target]][original_col_P_c[c_idx_target]]
                    val_A_permuted = A[P_r[r_idx_target]][P_c[c_idx_target]]
                    
                    if val_A_permuted != B[r_idx_target][c_idx_target]:
                        matches_B = False
                        break  # Mismatch found, no need to check further cells in this row
                if not matches_B:
                    break  # Mismatch found, no need to check further rows
            
            if matches_B:
                min_ops = current_total_ops # Update min_ops if a valid configuration is found with fewer ops

    if min_ops == float('inf'):
        sys.stdout.write("-1
")
    else:
        sys.stdout.write(str(min_ops) + "
")

# Entry point
if __name__ == '__main__':
    solve()