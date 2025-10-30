from typing import List
import heapq

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        """
        Finds the maximum sum of at most k elements from the grid,
        respecting row limits.
        """
        n = len(grid)
        # Handle the case of empty grid or rows, though constraints say n, m >= 1
        if n == 0 or len(grid[0]) == 0:
             return 0
             
        m = len(grid[0]) 

        # Collect the best possible elements from each row up to the limit
        # This list will store all elements that are candidates for selection
        available_elements = []
        for i in range(n):
            # Determine how many elements we can potentially take from this row
            num_to_take = min(m, limits[i])
            
            if num_to_take > 0:
                # Use heapq.nlargest to find the largest num_to_take elements in this row.
                # This is efficient as it avoids sorting the entire row if limits[i] is small.
                # heapq.nlargest directly finds the largest elements without a full sort.
                row_top_elements = heapq.nlargest(num_to_take, grid[i])
            else:
                # If limits[i] is 0, we can take no elements from this row
                row_top_elements = []

            # Add these top elements from the current row to our pool of available elements
            available_elements.extend(row_top_elements)

        # From the combined pool of available elements from all rows,
        # select the top k elements globally to maximize the sum.
        # heapq.nlargest(k, iterable) finds the k largest elements efficiently.
        # If k is larger than the number of available elements, it correctly returns
        # all available elements.
        top_k_elements = heapq.nlargest(k, available_elements)

        # The sum of these top k elements is the maximum possible sum under the constraints
        max_sum = sum(top_k_elements)

        return max_sum