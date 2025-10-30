from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the towers in descending order of their allowed maximum heights.
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        # Initialize the previous assigned height as infinity.
        prev = float('inf')
        
        # Greedily assign a height for each tower.
        for h in maximumHeight:
            # The allowed assignment is at most h and must be strictly less than the previous assigned height.
            assigned = min(h, prev - 1)
            # If we cannot assign a positive integer, it's impossible to assign distinct heights.
            if assigned < 1:
                return -1
            total_sum += assigned
            prev = assigned
        
        return total_sum