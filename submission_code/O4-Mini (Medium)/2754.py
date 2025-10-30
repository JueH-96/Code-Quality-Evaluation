from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll brute-force all non-empty subsets using bit masks.
        max_prod = -10**30  # A sufficiently small initial value.
        # Iterate over all non-empty subsets
        for mask in range(1, 1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod > max_prod:
                max_prod = prod
        return max_prod