from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        current = 0
        count = 0
        for num in nums:
            current += num
            if current == 0:
                count += 1
        return count