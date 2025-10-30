from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        all_elements = []
        for i in range(len(grid)):
            row = grid[i]
            limit = limits[i]
            # Sort the row in descending order and take up to 'limit' elements
            sorted_row = sorted(row, reverse=True)
            selected = sorted_row[:limit]
            all_elements.extend(selected)
        
        # Sort all collected elements in descending order
        all_elements.sort(reverse=True)
        
        # Take the top 'k' elements and sum them
        return sum(all_elements[:k])