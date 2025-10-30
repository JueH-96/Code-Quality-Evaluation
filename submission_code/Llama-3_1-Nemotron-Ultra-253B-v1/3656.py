from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        for k in range(len(nums) // 3 + 1):
            start = 3 * k
            sub = nums[start:]
            if len(set(sub)) == len(sub):
                return k
        return len(nums) // 3 + 1