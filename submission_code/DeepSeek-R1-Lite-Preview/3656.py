from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        k_max = (n + 2) // 3  # Ceiling of n/3
        for k in range(0, k_max + 1):
            s = 3 * k
            remaining = nums[s:]
            if len(set(remaining)) == len(remaining):
                return k
        return k_max