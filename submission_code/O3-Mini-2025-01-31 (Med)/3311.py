from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        boundary_count = 0  # to count how many times the ant returns to the boundary
        position = 0  # starting at boundary (position 0)
        for num in nums:
            position += num  # move the ant according to the step
            if position == 0:  # check if ant is exactly at the boundary after moving
                boundary_count += 1
        return boundary_count