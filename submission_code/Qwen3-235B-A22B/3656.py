from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_k = (n + 2) // 3
        for k in range(max_k + 1):
            start = 3 * k
            sub = nums[start:]
            if len(sub) == len(set(sub)):
                return k
        return max_k  # This line is a safeguard, though the problem ensures one valid k exists