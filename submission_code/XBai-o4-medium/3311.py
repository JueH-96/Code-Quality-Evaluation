from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        current_sum = 0
        count = 0
        for num in nums:
            current_sum += num
            if current_sum == 0:
                count += 1
        return count