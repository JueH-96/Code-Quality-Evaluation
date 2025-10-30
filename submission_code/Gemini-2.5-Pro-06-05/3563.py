from collections import defaultdict
from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        """
        Calculates the maximum score by selecting one or more cells from a grid,
        subject to the constraints that no two selected cells are in the same row,
        and all selected cell values are unique.
        """
        n = len(grid)
        m = len(grid[0])

        # Step 1: Pre-process the grid.
        # Create a mapping from each value to the set of rows it appears in.
        # This helps in quickly finding where a value can be chosen from.
        locations = defaultdict(set)
        for r in range(n):
            for c in range(m):
                locations[grid[r][c]].add(r)
        
        # Get a list of unique values, sorted in descending order.
        # Processing larger values first is a good heuristic.
        unique_vals = sorted(locations.keys(), reverse=True)
        num_unique_vals = len(unique_vals)
        
        # Memoization table to store results of subproblems.
        memo = {}

        def solve(k: int, row_mask: int) -> int:
            """
            Recursive solver with memoization (Top-Down DP).
            k: The index of the value in `unique_vals` being considered.
            row_mask: A bitmask representing rows that are already occupied.
            Returns the maximum score from the current state.
            """
            # Base case: If all unique values have been considered, no more score can be added.
            if k == num_unique_vals:
                return 0
            
            # If the result for this state is already computed, return it.
            state = (k, row_mask)
            if state in memo:
                return memo[state]

            val = unique_vals[k]
            
            # --- Choice 1: Skip the current value `val` ---
            # The score is determined by the best we can do with the remaining values.
            res = solve(k + 1, row_mask)

            # --- Choice 2: Pick the current value `val` ---
            # We can pick `val` if it exists in a row that is not yet occupied.
            for r in locations[val]:
                # Check if the r-th bit is 0, meaning row `r` is available.
                if not (row_mask & (1 << r)):
                    # Calculate the score if we pick `val` from row `r`.
                    # The new state has the next value index (k+1) and an updated mask.
                    pick_score = val + solve(k + 1, row_mask | (1 << r))
                    res = max(res, pick_score)
            
            # Memoize and return the result for the current state.
            memo[state] = res
            return res

        # Start the recursion from the first value (k=0) with no rows used (mask=0).
        return solve(0, 0)