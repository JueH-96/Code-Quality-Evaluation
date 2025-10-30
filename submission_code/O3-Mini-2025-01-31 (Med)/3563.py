from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n = len(grid)  # number of rows
        # Build a mapping from each value to a bitmask of rows in which it appears.
        # For a value v, bit i is set if grid[i] contains v.
        val_to_mask = {}
        for i, row in enumerate(grid):
            for v in row:
                if v not in val_to_mask:
                    val_to_mask[v] = 0
                val_to_mask[v] |= (1 << i)
        
        # We'll use a DP on the state of which rows have been used.
        # dp[mask] = maximum sum achieved when rows in 'mask' are already assigned.
        # There are at most 2^n states (with n<=10, that is 1024 states).
        Nstate = 1 << n
        # Initialize with -infinity for impossible states.
        dp = [-10**9] * Nstate
        dp[0] = 0
        
        # Iterate over each distinct value. Treat each value as an "item"
        # that we can choose at most once and assign to exactly one row (from those available).
        # Since picking orders don't matter, we can iterate in any order.
        for v, avail_mask in val_to_mask.items():
            # For each distinct value v, try to assign it to one free row that is available in avail_mask.
            new_dp = dp[:]  # copy current dp, we'll update transitions.
            for mask in range(Nstate):
                # Only consider states that are already achievable.
                if dp[mask] < 0:
                    continue
                # Determine free rows: rows not yet used.
                free = (~mask) & ((1 << n) - 1)
                # From free rows, get the rows in which v appears.
                possible_rows = free & avail_mask
                # For each row in possible_rows, set that bit in mask.
                while possible_rows:
                    # Get the lowest set bit:
                    row_bit = possible_rows & -possible_rows
                    new_mask = mask | row_bit
                    new_dp[new_mask] = max(new_dp[new_mask], dp[mask] + v)
                    # Remove the used bit:
                    possible_rows -= row_bit
            dp = new_dp
        
        # As we must select at least one cell, we ignore state 0 (which means no cell selected)
        return max(dp[mask] for mask in range(1, Nstate))