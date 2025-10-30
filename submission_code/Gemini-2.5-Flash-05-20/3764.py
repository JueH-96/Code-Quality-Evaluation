from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        m = len(grid[0]) # As per constraints, m is uniform for all rows.

        # Step 1: Create a list of all elements along with their row index.
        # This list will store tuples: (value, row_index)
        all_elements = []
        for r_idx in range(n):
            for c_idx in range(m):
                all_elements.append((grid[r_idx][c_idx], r_idx))

        # Step 2: Sort this list in descending order based on value.
        # Sorting tuples naturally sorts by the first element, then the second, etc.
        # We want descending order based on the value (first element of the tuple).
        all_elements.sort(key=lambda x: x[0], reverse=True)

        # Step 3: Initialize max_sum
        max_sum = 0

        # Step 4 & 5: Iterate through the sorted list and pick elements
        # The 'k' variable effectively acts as the remaining budget for total elements.
        # The 'limits' list is modified in-place to track remaining budget for each row.
        for value, r_idx in all_elements:
            # Check if we still have total capacity (k > 0)
            # AND if we still have row-specific capacity (limits[r_idx] > 0)
            if k > 0 and limits[r_idx] > 0:
                max_sum += value
                k -= 1           # Decrement total elements budget
                limits[r_idx] -= 1 # Decrement row-specific elements budget
            
            # Optimization: If k becomes 0, we have picked the maximum allowed
            # total number of elements. No need to process further elements,
            # even if they might be available per row limits.
            if k == 0:
                break
        
        # Step 6: Return the maximum sum
        return max_sum