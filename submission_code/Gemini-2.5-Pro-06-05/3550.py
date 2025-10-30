import heapq
from itertools import combinations
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        """
        Calculates the maximum sum of cell values for three non-attacking rooks.

        The "non-attacking" constraint means the three rooks must be in different
        rows and different columns.

        The approach is as follows:
        1. We need to select three cells (r1, c1), (r2, c2), (r3, c3) such that
           r1, r2, r3 are distinct and c1, c2, c3 are distinct, to maximize
           board[r1][c1] + board[r2][c2] + board[r3][c3].

        2. A brute-force check of all possible placements is too slow.
           A more efficient approach is to iterate through all combinations of three columns.
           The number of column combinations is nC3, which is manageable for n <= 100.

        3. For a fixed set of three columns {c1, c2, c3}, we need to find the
           optimal placement of three rooks in these columns, which involves selecting
           three distinct rows {r1, r2, r3}. A key insight is that for an optimal 
           solution, each chosen rook at (r, c) must be on a cell such that its value 
           is among the top 3 in that column `c`.

        4. This reduces the search space for rows significantly. For each of the
           three chosen columns, we only need to consider the top 3 rows.

        Algorithm:
        a. Pre-calculate the top 3 cell values and their row indices for each column.
           This takes O(m*n) time.
        b. Iterate through all combinations of 3 columns (c1, c2, c3). This is O(n^3).
        c. For each column combination, iterate through all 3*3*3 = 27 potential
           rook placements using the pre-calculated top 3 rows for each column.
        d. For each of these 27 combinations, check if the rows are distinct. If they are,
           calculate the sum and update the overall maximum sum.
        e. The total complexity will be O(m*n + n^3), which is efficient enough for the
           given constraints.
        """
        m, n = len(board), len(board[0])

        # Step 1: Pre-calculate top 3 values for each column.
        # top3_by_col[j] stores a list of (value, row_index) for column j.
        top3_by_col = []
        
        # Constraints guarantee m >= 3, so we can always find up to 3 candidates.
        num_candidates = 3
        
        for j in range(n):
            col_vals_with_indices = [(board[i][j], i) for i in range(m)]
            top3_by_col.append(heapq.nlargest(num_candidates, col_vals_with_indices))

        max_sum = -float('inf')

        # Step 2: Iterate over all combinations of 3 columns.
        # Constraints guarantee n >= 3.
        for cols in combinations(range(n), 3):
            c1, c2, c3 = cols
            
            # Step 3: Iterate through all candidate placements for this column triplet.
            list1 = top3_by_col[c1]
            list2 = top3_by_col[c2]
            list3 = top3_by_col[c3]
            
            for v1, r1 in list1:
                for v2, r2 in list2:
                    # Rooks must be in different rows.
                    if r1 == r2:
                        continue
                    for v3, r3 in list3:
                        if r3 != r1 and r3 != r2:
                            current_sum = v1 + v2 + v3
                            if current_sum > max_sum:
                                max_sum = current_sum
        
        return max_sum