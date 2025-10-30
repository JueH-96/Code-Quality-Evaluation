from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # List to hold the best candidates from each row within the limits.
        candidates = []
        
        for i, row in enumerate(grid):
            # Sort the row descending, then take at most limits[i] number of elements.
            if limits[i] > 0:
                sorted_row = sorted(row, reverse=True)
                # Only consider the top limits[i] elements (if limits[i] <= len(row))
                candidates.extend(sorted_row[:limits[i]])
        
        # Sort all the collected candidates in descending order.
        candidates.sort(reverse=True)
        
        # Sum the top k elements (if k is larger than available, it is guaranteed by the problem that k <= sum(limits)).
        return sum(candidates[:k])