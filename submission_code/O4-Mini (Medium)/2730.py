from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # prefix_or[i] = OR of nums[0..i-1]
        prefix_or = [0] * (n + 1)
        for i in range(n):
            prefix_or[i + 1] = prefix_or[i] | nums[i]
        # suffix_or[i] = OR of nums[i..n-1]
        suffix_or = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | nums[i]
        
        ans = 0
        # Try applying all k operations to each index i
        for i in range(n):
            # OR of all elements except nums[i]
            or_except_i = prefix_or[i] | suffix_or[i + 1]
            # Boost nums[i] by shifting left k times
            boosted = nums[i] << k
            # Combine
            current = or_except_i | boosted
            if current > ans:
                ans = current
        
        return ans