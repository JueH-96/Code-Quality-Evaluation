import sys
from itertools import permutations
from collections import Counter

def solve():
    """
    Reads grid data, solves the problem, and prints the result.
    """
    try:
        H_orig, W_orig = map(int, sys.stdin.readline().split())
        A_orig = [tuple(map(int, sys.stdin.readline().split())) for _ in range(H_orig)]
        B_orig = [tuple(map(int, sys.stdin.readline().split())) for _ in range(H_orig)]
    except (IOError, ValueError):
        # Handles cases with malformed or empty input.
        return

    # To optimize, we iterate over permutations of the smaller dimension.
    # If H > W, we transpose the grids and solve the equivalent problem.
    # The minimum number of swaps is invariant under transposition.
    if H_orig > W_orig:
        H, W = W_orig, H_orig
        A = [tuple(A_orig[r][c] for r in range(H_orig)) for c in range(W_orig)]
        B = [tuple(B_orig[r][c] for r in range(H_orig)) for c in range(W_orig)]
    else:
        H, W = H_orig, W_orig
        A, B = A_orig, B_orig

    def count_inversions(p):
        """Counts the minimum number of adjacent swaps to achieve permutation p."""
        inversions = 0
        n = len(p)
        for i in range(n):
            for j in range(i + 1, n):
                if p[i] > p[j]:
                    inversions += 1
        return inversions

    min_total_swaps = float('inf')
    
    # Pre-calculate B's row multiset for efficient comparison.
    b_row_counts = Counter(B)

    # Iterate through all permutations of column indices.
    col_indices = list(range(W))
    for col_perm in permutations(col_indices):
        
        col_swaps = count_inversions(col_perm)

        # Optimization: if column swaps alone are too high, prune this path.
        if col_swaps >= min_total_swaps:
            continue

        # Create the grid A_prime with permuted columns.
        a_prime_rows = []
        for i in range(H):
            new_row = tuple(A[i][col_perm[j]] for j in range(W))
            a_prime_rows.append(new_row)
        
        # Check if the multiset of rows in A_prime matches B's.
        if Counter(a_prime_rows) != b_row_counts:
            continue

        # If they match, a solution is possible. Find the row permutation cost.
        # p[i] = final position for row i of a_prime_rows.
        p = [-1] * H
        b_row_indices_used = [False] * H
        
        for i in range(H):
            current_a_prime_row = a_prime_rows[i]
            # Find the first unused matching row in B.
            for j in range(H):
                if not b_row_indices_used[j] and current_a_prime_row == B[j]:
                    p[i] = j
                    b_row_indices_used[j] = True
                    break
        
        row_swaps = count_inversions(p)
        
        total_swaps = col_swaps + row_swaps
        min_total_swaps = min(min_total_swaps, total_swaps)

    if min_total_swaps == float('inf'):
        print(-1)
    else:
        print(min_total_swaps)

# Run the solver.
solve()