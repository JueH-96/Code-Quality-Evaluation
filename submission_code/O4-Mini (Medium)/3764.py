from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # Edge case: no picks allowed
        if k <= 0:
            return 0
        
        # Collect all items as (value, row_index)
        items = []
        for i, row in enumerate(grid):
            for v in row:
                items.append((v, i))
        
        # Sort items by value descending
        items.sort(key=lambda x: x[0], reverse=True)
        
        # Track how many we've taken from each row
        row_used = [0] * len(grid)
        
        total = 0
        picked = 0
        
        # Greedily pick the largest remaining item that doesn't exceed row limit
        for value, r in items:
            if picked == k:
                break
            if row_used[r] < limits[r]:
                total += value
                row_used[r] += 1
                picked += 1
        
        return total