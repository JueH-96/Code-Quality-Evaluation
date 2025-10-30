from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # Collect up to limits[i] largest elements from each row
        candidates = []
        for row, lim in zip(grid, limits):
            if lim <= 0 or not row:
                continue
            # sort row in descending order
            row.sort(reverse=True)
            # take up to `lim` items from this row
            candidates.extend(row[:lim])
        
        # If no capacity or no candidates, answer is 0
        if k <= 0 or not candidates:
            return 0
        
        # Sort all candidates in descending order and sum the top k
        candidates.sort(reverse=True)
        return sum(candidates[:k])