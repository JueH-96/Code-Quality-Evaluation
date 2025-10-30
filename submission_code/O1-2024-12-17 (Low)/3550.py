from typing import List
from itertools import combinations

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        """
        We want to place exactly three non-attacking rooks (no two in the same row or column)
        so as to maximize the sum of the board values at those positions.

        Approach:
          1) We pick all combinations of 3 distinct rows in the board. Call them r1, r2, r3.
          2) For these three rows, we now need to choose exactly one distinct column per row,
             so that the sum of board[r1][c1] + board[r2][c2] + board[r3][c3] is maximized.
             This is maximum bipartite matching on a 3 x n matrix (3 rows vs n columns),
             where the "cost" for row i choosing column j is board[r_i][j].
          3) We can solve this 3 x n maximum matching problem efficiently using a modified
             Hungarian algorithm for rectangular matrices (3 rows, n columns). The time is
             O(3^2 * n + 3*n^2) ~ O(n^2), which is acceptable for n up to 100.
          4) We keep track of the global maximum over all row-triplets.

        This solution should run in reasonable time for m, n <= 100.
        """

        # Precompute all combinations of 3 distinct rows
        row_combos = list(combinations(range(len(board)), 3))

        # We'll implement a helper to get maximum cost matching for a 3 x n matrix.
        # The matrix will be cost[row][col], row in [0..2], col in [0..n-1].
        def hungarian_3_by_n_max(cost3xn: List[List[int]]) -> int:
            """
            Return the maximum sum of choosing one distinct column for each of the 3 rows.
            We'll implement a Hungarian-like algorithm for a rectangular cost matrix (3 x n).
            Since we have only 3 rows, we can adapt the Hungarian method (which is typically
            laid out for square matrices) with minor modifications. We'll treat it as:
               - #workers (rows) = 3
               - #jobs (columns) = n
            We'll find an assignment of each row to exactly one distinct column that maximizes sum.
            
            High-level steps for the "max cost" version on a 3 x n:
              1) Convert to a "min cost" problem by subtracting each cost from a large offset.
              2) Apply the Hungarian (Kuhn-Munkres) in a standard rectangular form.
              3) Convert back the result to max sum.

            For numeric stability, we can find the overall max value in cost3xn and offset
            everything so that we do min-cost Hungarian on (maxVal - cost).
            """
            # Number of rows (fixed = 3), number of cols = n
            m = 3
            n = len(cost3xn[0]) if m > 0 else 0
            if n < 3:
                # If there are fewer than 3 columns, no valid way to pick 3 distinct columns => 
                # Actually we can't place 3 rooks if n < 3. Return some minimum large negative.
                return float('-inf')

            # Find the maximum entry to transform for min-cost
            max_val = max(max(row) for row in cost3xn)
            # Build the cost matrix for min-cost Hungarian: costMin[i][j] = offset - cost[i][j]
            # where offset = max_val + something so that costMin remains non-negative.
            # However, since Hungarian can handle non-negative but also handles 0-based is typical.
            # We can simply do offset = max_val if cost could be negative. This is fine as Python
            # handles large integers, but we should be mindful of possible negative costs up to -1e9.
            # We'll do an offset to ensure positiveness, e.g. offset = max_val + 10^9 to handle negative.
            # But that might blow up. Actually, Python int is unbounded, so we can do offset = max_val + 10^10 safely.
            offset = max_val + 10**10
            costMin = [[offset - cost3xn[i][j] for j in range(n)] for i in range(m)]
            
            # Now we run a standard Hungarian (Kuhn-Munkres) for a rectangular matrix of size m x n
            # to find the min-cost matching covering the m rows. We'll store the assignment row->col.
            # Hungarian reference complexity: O(m*n*min(m,n)). Here min(m,n)=3, so O(3*m*n)=O(m*n).
            # We'll do an efficient but standard implementation for rectangular shape.

            # u and v will store the potential for rows and columns (for Hungarian),
            # p[j] = which row is matched to column j, way[j] = the column's "link" in the adjacency sense.
            u = [0]*(m+1)
            v = [0]*(n+1)
            p = [0]*(n+1)
            way = [0]*(n+1)

            for i in range(1, m+1):
                p[0] = i
                j0 = 0  # column matched with row i
                minv = [float('inf')]*(n+1)
                used = [False]*(n+1)
                while True:
                    used[j0] = True
                    i0 = p[j0]
                    j1 = 0
                    delta = float('inf')
                    for j in range(1, n+1):
                        if not used[j]:
                            cur = costMin[i0-1][j-1] - u[i0] - v[j]
                            if cur < minv[j]:
                                minv[j] = cur
                                way[j] = j0
                            if minv[j] < delta:
                                delta = minv[j]
                                j1 = j
                    for j in range(n+1):
                        if used[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            minv[j] -= delta
                    j0 = j1
                    if p[j0] == 0:
                        break
                while True:
                    j1 = way[j0]
                    p[j0] = p[j1]
                    j0 = j1
                    if j0 == 0:
                        break
            
            # p[j] is the row matched to column j
            # We only need columns j=1..n that actually matched to row != 0
            # The cost of that assignment in "min" form is sum of costMin[row-1][j-1].
            # We want the sum of the original cost => transform back.
            # We only have 3 assigned columns (since each of the 3 rows is matched exactly once).
            # So let's collect which column is assigned to row 1, row 2, row 3.
            # We'll do rowAssign[r] = the column assigned to that row (0-based).
            rowAssign = [-1]*m
            for j in range(1, n+1):
                r = p[j]
                if 1 <= r <= m:
                    rowAssign[r-1] = j-1

            # Now compute the sum of the original cost using rowAssign
            total_sum = 0
            for r in range(m):
                c = rowAssign[r]
                total_sum += cost3xn[r][c]

            return total_sum

        max_sum = float('-inf')

        # Try all row triplets
        for (r1, r2, r3) in row_combos:
            # Build the 3 x n cost matrix
            cost_matrix = [
                board[r1],
                board[r2],
                board[r3]
            ]
            best_for_these_rows = hungarian_3_by_n_max(cost_matrix)
            if best_for_these_rows > max_sum:
                max_sum = best_for_these_rows

        return max_sum