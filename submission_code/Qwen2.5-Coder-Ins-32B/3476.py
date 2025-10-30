from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = [0, 0, 0]
        for num in nums:
            count[num % 3] += 1
        return len(nums) - max(count)