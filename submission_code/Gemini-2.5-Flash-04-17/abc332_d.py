import sys
import itertools
from collections import Counter

def count_inversions(perm):
    """Counts inversions in a permutation using O(N^2) naive approach."""
    n = len(perm)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return inversions

def solve():
    H, W = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    B = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]

    min_total_inversions = float('inf')

    # Iterate through all possible permutations of rows P.
    # Let P be a permutation of {0, ..., H-1}.
    # If we apply row permutation P, the row that was originally at index P[i] in grid A
    # will move to target row index i in the resulting grid.
    # The cost of this row permutation is the number of inversions in P.
    # This is because adjacent swaps can generate any permutation, and the minimum number
    # of adjacent swaps to achieve a permutation P is the number of inversions in P.
    # P[i] represents the original row index that should end up at target row index i.

    for p_tuple in itertools.permutations(range(H)):
        P = list(p_tuple) # P[i] is the original row index that becomes target row index i.
        inv_P = count_inversions(P) # Number of swaps to achieve this row order.

        # Construct the matrix A_P where row i is the original row P[i] from A.
        # This is the state of the grid after applying the row permutation P.
        A_P_rows = [A[P[i]] for i in range(H)]

        # Now, we need to check if the columns of A_P_rows can be permuted to match grid B.
        # This means A_P_rows with its columns permuted by Q must equal B.
        # Let Q be a permutation of {0, ..., W-1}. If we apply column permutation Q,
        # the element at (i, k) in A_P_rows moves to (i, j) where j is the new column index.
        # The operation swaps column j and j+1. If we want the column that was originally at index k
        # in A_P_rows to end up at target column index j in B, the permutation Q maps original column
        # index k to target column index j. Q[k] = j.
        # The minimum number of column swaps is the number of inversions in this permutation Q.
        # The element at A_P_rows[i][k] must end up at B[i][j] where Q[k] = j.
        # So, A_P_rows[i][k] = B[i][Q[k]] for all i, k.
        # This means the k-th column of A_P_rows must be equal to the Q[k]-th column of B.

        # Let A_P_cols be the list of column vectors of A_P_rows.
        A_P_cols = [tuple(A_P_rows[i][k] for i in range(H)) for k in range(W)]

        # Let B_cols be the list of column vectors of B.
        B_cols = [tuple(B[i][j] for i in range(H)) for j in range(W)]

        # For A_P_rows to be transformable into B by column permutation, the multiset of column
        # vectors in A_P_cols must be equal to the multiset of column vectors in B_cols.
        if Counter(A_P_cols) != Counter(B_cols):
            continue # This row permutation P cannot lead to B

        # If multisets match, a column permutation Q exists such that A_P_cols[k] == B_cols[Q[k]] for all k.
        # We need to find this permutation Q = [Q[0], Q[1], ..., Q[W-1]] that maps original column
        # index k in A_P_rows to target column index Q[k] in B, such that A_P_cols[k] == B_cols[Q[k]].
        # To minimize inversions in Q, we should match columns with the same vector value.
        # If multiple columns have the same vector value, we match them based on their original and target indices.
        # The original column indices k for a vector v form a set {k | A_P_cols[k] == v}.
        # The target column indices j for a vector v form a set {j | B_cols[j] == v}.
        # We need to map the set of original indices to the set of target indices for each vector v.
        # To minimize inversions, the i-th smallest original index for vector v must map to the i-th smallest target index for vector v.

        vec_to_orig_indices = {}
        for k in range(W):
            vec = A_P_cols[k]
            if vec not in vec_to_orig_indices: vec_to_orig_indices[vec] = []
            vec_to_orig_indices[vec].append(k)

        vec_to_target_indices = {}
        for j in range(W):
            vec = B_cols[j]
            if vec not in vec_to_target_indices: vec_to_target_indices[vec] = []
            vec_to_target_indices[vec].append(j)

        # Construct the column permutation Q. Q[original_k] = target_j means original column index k
        # in A_P_rows moves to target column index j in B.
        Q_perm = [None] * W # Q_perm[original_k] = target_j

        for vec in vec_to_orig_indices:
            orig_indices = sorted(vec_to_orig_indices[vec])
            target_indices = sorted(vec_to_target_indices[vec])
            # For each i from 0 to len(orig_indices)-1, the original column `orig_indices[i]`
            # (which has vector `vec`) must end up at target column `target_indices[i]`
            # (which also has vector `vec`).
            # The permutation Q maps original index `orig_k` to target index `target_j`.
            for i in range(len(orig_indices)):
                 Q_perm[orig_indices[i]] = target_indices[i]

        inv_Q = count_inversions(Q_perm) # Number of swaps for this column permutation.

        # Update minimum total inversions
        total_inversions = inv_P + inv_Q
        min_total_inversions = min(min_total_inversions, total_inversions)

    if min_total_inversions == float('inf'):
        print(-1)
    else:
        print(min_total_inversions)

solve()