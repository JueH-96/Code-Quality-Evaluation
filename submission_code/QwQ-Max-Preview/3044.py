from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        rightmost = {}
        n = len(nums)
        for i in reversed(range(n)):
            if nums[i] not in rightmost:
                rightmost[nums[i]] = i
        max_step = 0
        for x in range(1, k + 1):
            idx = rightmost[x]
            current_step = n - idx
            if current_step > max_step:
                max_step = current_step
        return max_step