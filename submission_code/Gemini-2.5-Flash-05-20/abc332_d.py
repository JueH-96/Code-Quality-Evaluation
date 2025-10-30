import itertools

def count_inversions(arr):
    """
    Counts the number of inversions in a permutation.
    An inversion is a pair (i, j) such that i < j and arr[i] > arr[j].
    The minimum number of adjacent swaps required to transform a sorted
    sequence into a given permutation is equal to the number of inversions
    in that permutation.
    """
    inversions = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions

def solve():
    H, W = map(int, input().split())

    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))

    B = []
    for _ in range(H):
        B.append(list(map(int, input().split())))

    min_total_ops = float('inf')

    # `row_original_indices` and `col_original_indices` represent the initial
    # order of rows and columns (0-indexed).
    row_original_indices = list(range(H))
    col_original_indices = list(range(W))

    # Iterate through all possible permutations of row indices.
    # A permutation `p_row` means that `B[r_target]` corresponds to `A[p_row[r_target]]`.
    for p_row_tuple in itertools.permutations(row_original_indices):
        p_row = list(p_row_tuple) # Convert tuple to list for convenient indexing

        # Iterate through all possible permutations of column indices.
        # A permutation `p_col` means that `B[c_target]` corresponds to `A[p_col[c_target]]`.
        for p_col_tuple in itertools.permutations(col_original_indices):
            p_col = list(p_col_tuple) # Convert tuple to list for convenient indexing

            # Check if applying these permutations to A makes it identical to B.
            is_match = True
            for r_target in range(H): # Iterate through target row indices (0 to H-1) in B
                for c_target in range(W): # Iterate through target column indices (0 to W-1) in B
                    # The value at B[r_target][c_target] must be equal to
                    # the value from A[original_row_index][original_col_index], where
                    # original_row_index is p_row[r_target]
                    # original_col_index is p_col[c_target]
                    if A[p_row[r_target]][p_col[c_target]] != B[r_target][c_target]:
                        is_match = False
                        break # No match for this (p_row, p_col) pair
                if not is_match:
                    break
            
            if is_match:
                # If a match is found, calculate the total number of operations.
                # The total operations = inversions for the row permutation + inversions for the column permutation.
                current_ops = count_inversions(p_row) + count_inversions(p_col)
                min_total_ops = min(min_total_ops, current_ops)

    if min_total_ops == float('inf'):
        print(-1) # No combination of permutations made A identical to B
    else:
        print(min_total_ops)

solve()