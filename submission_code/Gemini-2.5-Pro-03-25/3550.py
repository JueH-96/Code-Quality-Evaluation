import math
import itertools
import heapq
from typing import List

class Solution:
    """
    Finds the maximum sum of values for three non-attacking rooks on a chessboard.

    The problem requires placing three rooks on an m x n board such that no two rooks
    share the same row or column, and the sum of the values of the cells they occupy
    is maximized.

    Let the positions of the three rooks be (r1, c1), (r2, c2), and (r3, c3).
    The constraints are: r1, r2, r3 must be distinct row indices, and c1, c2, c3 
    must be distinct column indices.
    We want to maximize the sum achieved by one of the 3! = 6 possible assignments 
    of the chosen columns to the chosen rows. For example, one possible sum is 
    board[r1][c1] + board[r2][c2] + board[r3][c3].

    A naive brute-force approach checking all combinations of three cells or all
    valid rook placements is computationally too expensive given the constraints 
    m, n <= 100.

    The chosen approach optimizes this by iterating through combinations of the 
    smaller dimension cubically, and using a top-k heuristic for the larger dimension 
    to keep the complexity manageable.
    
    1. We ensure that the number of columns (n') is less than or equal to the 
       number of rows (m') by transposing the board if necessary (m' >= n').
    2. We iterate through all combinations of 3 distinct columns (c1, c2, c3). 
       This takes O(n'^3) time.
    3. For each fixed combination of columns, we need to find the best combination 
       of 3 distinct rows (r1, r2, r3) that maximizes the sum over the 6 possible 
       assignments (permutations) of columns to rows.
    4. To avoid iterating through all O(m'^3) row combinations, we use a heuristic:
       - Find the top `k` rows with the largest values in column c1.
       - Find the top `k` rows with the largest values in column c2.
       - Find the top `k` rows with the largest values in column c3.
       (Here, `k` is a small constant, e.g., 5. If the number of rows m' < k, we use m').
       - Collect the unique indices of these top rows (at most 3k rows). This forms 
         our set of `candidate_rows`. The rationale is that the optimal rows are likely 
         to be among those that have high values in at least one of the selected columns.
       - Iterate through all combinations of 3 distinct rows from this `candidate_rows` set. 
         This takes O(k^3) time per column combination.
       - For each combination of 3 rows and 3 columns, calculate the maximum sum achievable 
         over the 6 permutations.
       - Keep track of the overall maximum sum found.

    The overall time complexity is O(n'^3 * (m' + k^3)), where n' = min(m, n) and 
    m' = max(m, n). Since k is a small constant, this simplifies to O(min(m,n)^3 * max(m,n)).
    Given m, n <= 100, the maximum number of operations is roughly 100^3 * 100 = 10^8, 
    which should be feasible within typical time limits.
    """
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        # board_ref will point to the board we operate on (original or transposed).
        # m_prime and n_prime will store the dimensions of board_ref.
        board_ref = board
        m_prime, n_prime = m, n
        
        # Ensure n_prime <= m_prime by transposing if m < n.
        if m < n:
            # Transpose the board
            new_board = [[0] * m for _ in range(n)]
            for r in range(m):
                for c in range(n):
                    new_board[c][r] = board[r][c]
            board_ref = new_board
            # Swap dimensions: the transposed board has n rows and m columns.
            m_prime, n_prime = n, m 
            # Now, m_prime >= n_prime

        # Initialize the maximum sum found so far to negative infinity.
        max_total_sum = -float('inf')
        
        # k: number of top rows to consider for each column.
        # Must be at least 3 to be able to select 3 distinct rows.
        # k=5 is chosen as a heuristic value providing a safety margin.
        # If m_prime (number of rows) is smaller than k, we only consider m_prime rows.
        k = min(5, m_prime) 

        # Get a list of column indices [0, 1, ..., n_prime-1]
        cols = list(range(n_prime))
        
        # Iterate through all combinations of 3 distinct columns (O(n_prime^3) combinations)
        for col_combo in itertools.combinations(cols, 3):
            c1, c2, c3 = col_combo
            
            # --- Find candidate rows based on top-k values in selected columns ---
            # For each selected column, find the indices of the k rows with the largest values.
            # Using heapq.nlargest is efficient for finding top k elements. 
            # Complexity is O(m_prime log k) or O(m_prime) if k is small relative to m_prime.
            
            # Store tuples (value, row_index) for sorting/selection
            rows_with_vals_c1 = [(board_ref[r][c1], r) for r in range(m_prime)]
            rows_with_vals_c2 = [(board_ref[r][c2], r) for r in range(m_prime)]
            rows_with_vals_c3 = [(board_ref[r][c3], r) for r in range(m_prime)]

            # Get the lists of (value, row_index) for the top k rows in each column
            top_rows_c1 = heapq.nlargest(k, rows_with_vals_c1)
            top_rows_c2 = heapq.nlargest(k, rows_with_vals_c2)
            top_rows_c3 = heapq.nlargest(k, rows_with_vals_c3)

            # Collect the unique indices of these top rows into a set.
            # The size of this set is at most 3k.
            candidate_rows = set()
            for _, r in top_rows_c1: candidate_rows.add(r)
            for _, r in top_rows_c2: candidate_rows.add(r)
            for _, r in top_rows_c3: candidate_rows.add(r)

            # Convert the set of candidate row indices to a list for iteration.
            candidate_rows_list = list(candidate_rows)
            
            # We need at least 3 distinct rows to form a valid placement.
            if len(candidate_rows_list) < 3:
                # If there are fewer than 3 unique candidate rows (e.g., if m_prime < 3), 
                # we cannot form a combination of 3 distinct rows from this set.
                continue 

            # Store the maximum sum found for this specific combination of columns.
            max_sum_for_cols = -float('inf')

            # --- Iterate through combinations of 3 rows from the candidates ---
            # The number of combinations is O(k^3) since |candidate_rows_list| <= 3k.
            for row_combo in itertools.combinations(candidate_rows_list, 3):
                r1, r2, r3 = row_combo
                
                # --- Calculate the maximum sum over 6 permutations ---
                # Get the values at the 9 intersection points of the chosen rows and columns.
                val_r1c1, val_r1c2, val_r1c3 = board_ref[r1][c1], board_ref[r1][c2], board_ref[r1][c3]
                val_r2c1, val_r2c2, val_r2c3 = board_ref[r2][c1], board_ref[r2][c2], board_ref[r2][c3]
                val_r3c1, val_r3c2, val_r3c3 = board_ref[r3][c1], board_ref[r3][c2], board_ref[r3][c3]

                # Calculate the sum for each of the 3! = 6 permutations.
                p1 = val_r1c1 + val_r2c2 + val_r3c3  # Assignment: (r1,c1), (r2,c2), (r3,c3)
                p2 = val_r1c1 + val_r2c3 + val_r3c2  # Assignment: (r1,c1), (r2,c3), (r3,c2)
                p3 = val_r1c2 + val_r2c1 + val_r3c3  # Assignment: (r1,c2), (r2,c1), (r3,c3)
                p4 = val_r1c2 + val_r2c3 + val_r3c1  # Assignment: (r1,c2), (r2,c3), (r3,c1)
                p5 = val_r1c3 + val_r2c1 + val_r3c2  # Assignment: (r1,c3), (r2,c1), (r3,c2)
                p6 = val_r1c3 + val_r2c2 + val_r3c1  # Assignment: (r1,c3), (r2,c2), (r3,c1)
                
                # Find the maximum sum among the 6 permutations for this row/column combination.
                current_max_perm_sum = max(p1, p2, p3, p4, p5, p6)
                
                # Update the maximum sum found so far for the current fixed columns.
                max_sum_for_cols = max(max_sum_for_cols, current_max_perm_sum)

            # After checking all row combinations for the fixed columns, update the overall max sum.
            # Check if max_sum_for_cols was updated (i.e., if at least one valid row combo existed).
            if max_sum_for_cols != -float('inf'):
                 max_total_sum = max(max_total_sum, max_sum_for_cols)

        # The constraints m, n >= 3 guarantee that it's always possible to place 3 non-attacking rooks.
        # Therefore, max_total_sum should have been updated from its initial -infinity value.
        return max_total_sum