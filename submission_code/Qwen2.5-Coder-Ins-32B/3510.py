from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        # Use a set to keep track of used heights
        used_heights = set()
        total_sum = 0
        
        for max_height in maximumHeight:
            # Find the largest possible unique height for the current tower
            for height in range(max_height, 0, -1):
                if height not in used_heights:
                    used_heights.add(height)
                    total_sum += height
                    break
            else:
                # If no unique height is found, return -1
                return -1
        
        return total_sum