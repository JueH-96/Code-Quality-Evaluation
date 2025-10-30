class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        import math
        n = len(fruits)
        # Special case
        if n == 1:
            return fruits[0][0]
        
        # 1) Child A always takes the main diagonal (0,0)->(1,1)->...->(n-1,n-1).
        #    Those fruits are guaranteed collected exactly once by A.
        #    Sum them up and then set them to 0 so B and C do not collect them.
        diag_sum = 0
        for i in range(n):
            diag_sum += fruits[i][i]
            fruits[i][i] = 0
        
        # 2) Compute the maximum fruits child B can collect ignoring collisions,
        #    from (0, n-1) down to (n-1, n-1) with exactly n-1 "down" steps
        #    and column moves in {-1, 0, +1}.
        #    We'll use a DP table BDP[r][c] = best sum if B ends at (r,c).
        NEG_INF = -10**15
        BDP = [[NEG_INF]*n for _ in range(n)]
        # Starting position for B is (0, n-1)
        BDP[0][n-1] = fruits[0][n-1]
        
        for r in range(1, n):
            prev_row = BDP[r-1]
            curr_row = BDP[r]
            for c in range(n):
                if prev_row[c] != NEG_INF:
                    # same column
                    val = prev_row[c] + fruits[r][c]
                    if val > curr_row[c]:
                        curr_row[c] = val
                if c > 0 and prev_row[c-1] != NEG_INF:
                    # coming from column c-1
                    val = prev_row[c-1] + fruits[r][c]
                    if val > curr_row[c]:
                        curr_row[c] = val
                if c < n-1 and prev_row[c+1] != NEG_INF:
                    # coming from column c+1
                    val = prev_row[c+1] + fruits[r][c]
                    if val > curr_row[c]:
                        curr_row[c] = val
        
        Bmax = BDP[n-1][n-1]  # best sum for child B
        
        # 3) Compute the maximum fruits child C can collect ignoring collisions,
        #    from (n-1, 0) to (n-1, n-1) in exactly n-1 "right" steps
        #    and row moves in {-1, 0, +1}.
        #    We'll use CDP[c][r] = best sum if C is at (r,c).
        CDP = [[NEG_INF]*n for _ in range(n)]
        # Starting position for C is (n-1, 0)
        CDP[0][n-1] = fruits[n-1][0]
        
        for c in range(1, n):
            prev_col = CDP[c-1]
            curr_col = CDP[c]
            for r in range(n):
                if prev_col[r] != NEG_INF:
                    # same row
                    val = prev_col[r] + fruits[r][c]
                    if val > curr_col[r]:
                        curr_col[r] = val
                if r > 0 and prev_col[r-1] != NEG_INF:
                    # coming from row r-1
                    val = prev_col[r-1] + fruits[r][c]
                    if val > curr_col[r]:
                        curr_col[r] = val
                if r < n-1 and prev_col[r+1] != NEG_INF:
                    # coming from row r+1
                    val = prev_col[r+1] + fruits[r][c]
                    if val > curr_col[r]:
                        curr_col[r] = val
        
        Cmax = CDP[n-1][n-1]  # best sum for child C
        
        # 4) In this setup, the cell (n-1,n-1) was originally on the diagonal,
        #    so we have already given its fruits to child A (and subsequently
        #    set it to 0).  That means child B and C each see that corner as 0
        #    and do not re-collect it.  Hence there is no over-count to subtract.
        
        return diag_sum + Bmax + Cmax