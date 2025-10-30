from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        # Current position of the ant relative to the boundary
        pos = 0
        # Number of times the ant returns to the boundary (position 0)
        count = 0
        
        for step in nums:
            pos += step       # Move the ant
            if pos == 0:      # Check if it is back at the boundary
                count += 1
                
        return count