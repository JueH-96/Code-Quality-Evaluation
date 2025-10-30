class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        # 1) Child 1 must take the main diagonal (only way to make exactly n-1 moves and get from (0,0) to (n-1,n-1))
        diag_sum = sum(fruits[i][i] for i in range(n))
        # 2) Child 3 (from bottom‐left) can pick exactly one fruit cell in each column.
        #    Staying on row n-1 (the bottom row) for all columns 0..n-1 is feasible
        #    (moves are (−1,1),(0,1),(1,1) and you start/end at row n-1 so net vertical change zero).
        #    That collects all fruits on the bottom row.
        bottom_sum = sum(fruits[n-1][j] for j in range(n))
        # 3) Child 2 (from top‐right) can pick exactly one fruit cell in each row.
        #    Staying on column n-1 (the rightmost column) for all rows 0..n-1 is feasible
        #    (moves are (1,−1),(1,0),(1,1) and you start/end at col n-1 so net horizontal change zero).
        #    That collects all fruits on the right column.
        right_sum = sum(fruits[i][n-1] for i in range(n))
        # Now we've double‐counted the three corner cells:
        #   (0,n-1), (n-1,0), and (n-1,n-1) each appears in two of the sums above.
        # We need to subtract them once so each fruit is counted exactly once.
        correction = fruits[0][n-1] + fruits[n-1][0] + fruits[n-1][n-1]
        return diag_sum + bottom_sum + right_sum - correction