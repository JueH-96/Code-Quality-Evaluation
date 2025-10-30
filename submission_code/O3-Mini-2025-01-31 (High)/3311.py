from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        pos = 0
        count = 0
        for move in nums:
            pos += move
            if pos == 0:  # Check if the ant is exactly on the boundary after the move
                count += 1
        return count