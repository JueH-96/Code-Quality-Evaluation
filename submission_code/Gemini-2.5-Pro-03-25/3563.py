import collections
from typing import List

class Solution:
    """
    Solves the problem of finding the maximum score by selecting cells from a grid
    such that no two selected cells are in the same row and all selected cell values are unique.
    Uses dynamic programming based on the values present in the grid.
    The state dp[mask] represents the maximum score using rows indicated by mask,
    considering a subset of values processed so far.
    """
    def maxScore(self, grid: List[List[int]]) -> int:
        """
        Calculates the maximum achievable score.

        Args:
            grid: A 2D list of positive integers representing the grid.

        Returns:
            The maximum score satisfying the conditions.
        """
        m = len(grid)
        # Basic grid dimension checks. Constraints state m >= 1.
        if m == 0:
            return 0
        n = len(grid[0])
        # Constraints state n >= 1.
        if n == 0:
             return 0

        # Step 1: Precompute which rows contain each unique value.
        # `rows_containing_value`: dictionary mapping value -> set of row indices where the value appears.
        # This helps quickly find candidate rows for a given value.
        rows_containing_value = collections.defaultdict(set)
        # `unique_values`: set of all unique values present in the grid.
        unique_values = set()
        for r in range(m):
            # Using a set for the current row avoids redundant checks if a value appears multiple times in the same row.
            row_unique_values = set(grid[r]) 
            for value in row_unique_values:
                 # Map the value to the row index it appears in.
                 rows_containing_value[value].add(r)
                 # Add the value to the global set of unique values.
                 unique_values.add(value) 

        # Step 2: Sort the unique values. Processing values in increasing order (or any fixed order)
        # ensures that the DP state transitions are correctly calculated based on previously processed values.
        sorted_unique_values = sorted(list(unique_values))
        
        # Step 3: Initialize the DP table.
        # `dp_prev[mask]` stores the maximum score achievable using a subset of rows represented by `mask`,
        # considering only values processed *before* the current value `v`.
        # The size of the DP table is 2^m, covering all possible subsets of rows.
        # Initialize with -1 to represent unreachable states.
        dp_prev = [-1] * (1 << m)
        # Base case: A score of 0 is achieved with mask 0 (selecting no rows and no values).
        dp_prev[0] = 0
        
        # Step 4: Iterate through each unique value `v` present in the grid, in sorted order.
        for v in sorted_unique_values:
            # `dp_curr` will store the DP results after considering the current value `v`.
            # Initialize `dp_curr` by copying `dp_prev`. This accounts for the possibility of *not* selecting the current value `v`.
            dp_curr = dp_prev[:] 
            
            # Get the set of rows where the current value `v` appears.
            possible_rows = rows_containing_value[v]
                
            # Iterate through all possible masks (representing subsets of rows).
            for mask in range(1 << m):
                # For the current mask, consider assigning value `v` to one of the rows `r` included in the mask.
                for r in possible_rows:
                    # Check if row `r` is included in the current `mask`. The k-th bit of mask is 1 if row k is included.
                    if (mask >> r) & 1:
                        # If row `r` is set in `mask`, it means this state could potentially represent a selection
                        # where value `v` is assigned to row `r`.
                        # To calculate the score for this possibility, we need the score of the state *before* `v` was assigned to `r`.
                        # This previous state corresponds to the mask `prev_mask` which excludes row `r`.
                        prev_mask = mask ^ (1 << r)
                        
                        # Check if the state `prev_mask` was reachable using values strictly less than `v`.
                        # `dp_prev[prev_mask]` holds the max score for `prev_mask` using values processed before `v`.
                        if dp_prev[prev_mask] != -1:
                            # If `prev_mask` was reachable, calculate the potential score by adding `v`.
                            score = dp_prev[prev_mask] + v
                            # Update the score for the current state `mask` in `dp_curr`
                            # if assigning `v` to row `r` yields a better score than previously found for `mask`.
                            dp_curr[mask] = max(dp_curr[mask], score)
            
            # After considering all masks and possibilities for assigning `v`,
            # update `dp_prev` to `dp_curr`. `dp_prev` now holds the results after processing `v`.
            dp_prev = dp_curr

        # Step 5: Find the maximum score in the final DP table.
        # After iterating through all unique values, `dp_prev` contains the maximum scores for each possible subset of rows.
        # The overall maximum score is the maximum value found in the `dp_prev` array.
        # Since `dp_prev[0]` is always 0, `max(dp_prev)` will return at least 0.
        return max(dp_prev)