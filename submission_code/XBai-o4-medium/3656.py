from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_k = (n + 2) // 3
        for k in range(0, max_k + 1):
            start = 3 * k
            if start >= n:
                return k
            subarray = nums[start:]
            if len(subarray) == len(set(subarray)):
                return k
        return max_k  # This line is a fallback and theoretically unreachable