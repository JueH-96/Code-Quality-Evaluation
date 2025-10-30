from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        current_position = 0
        count = 0
        for num in nums:
            current_position += num
            if current_position == 0:
                count += 1
        return count