from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        sorted_heights = sorted(maximumHeight, reverse=True)
        
        max_assigned_height = 0
        total_sum = 0
        
        # Iterate through the sorted maximum heights
        for height in sorted_heights:
            # Assign the minimum of height and max_assigned_height + 1
            assigned_height = min(height, max_assigned_height + 1)
            # If the assigned height is not greater than max_assigned_height, it's impossible
            if assigned_height <= max_assigned_height:
                return -1
            # Update max_assigned_height and add to total sum
            max_assigned_height = assigned_height
            total_sum += assigned_height
        
        return total_sum