class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # 1) Collect all distinct values in the matrix.
        distinct_vals = set()
        for row in grid:
            distinct_vals.update(row)
        distinct_vals = sorted(distinct_vals)  # ascending order

        n = len(grid)   # number of rows
        # 2) For each distinct value, build a bitmask indicating which rows contain it.
        row_masks = []
        for val in distinct_vals:
            mask = 0
            for r in range(n):
                if val in grid[r]:
                    mask |= (1 << r)
            row_masks.append(mask)

        # 3) We'll do DP over (index of distinct value) vs (bitmask of used rows).
        # dp[used] = maximum sum achievable using some subset of the values processed so far
        #            where 'used' indicates which rows have been used.
        M = len(distinct_vals)
        dp = [0] * (1 << n)  # initially for i=0 (no values processed)

        # 4) Iterate over the distinct values
        for i in range(M):
            val = distinct_vals[i]
            mask_for_val = row_masks[i]
            dp_next = dp[:]  # copy current dp to dp_next for the next state

            # Consider picking this value in one of the rows that contains it
            for used in range(1 << n):
                current_sum = dp[used]
                # find free rows that actually contain 'val'
                free_rows_mask = (~used) & mask_for_val
                # try to place 'val' in one of these free rows
                r = free_rows_mask
                while r > 0:
                    # isolate the lowest set bit (row)
                    pick_row = r & (-r)
                    used2 = used | pick_row
                    dp_next[used2] = max(dp_next[used2], current_sum + val)
                    # clear the lowest set bit so we can check the next
                    r = r & (r - 1)

            dp = dp_next

        # 5) Since all values are positive, the best answer will never be with used=0 if we must pick ≥1 cell.
        #    However, we still compute the max overall (excluding used=0 if desired). 
        return max(dp)