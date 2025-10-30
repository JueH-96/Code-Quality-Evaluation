from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_k = n // 3 + (n % 3 > 0)
        for k in range(max_k + 1):
            remaining = nums[3 * k:]
            if len(set(remaining)) == len(remaining):
                return k
        return -1  # This line should theoretically never be reached