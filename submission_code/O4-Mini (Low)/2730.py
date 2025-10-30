from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # prefix_or[i] = OR of nums[0..i-1]
        prefix_or = [0] * (n + 1)
        for i in range(n):
            prefix_or[i+1] = prefix_or[i] | nums[i]
        
        # suffix_or[i] = OR of nums[i..n-1]
        suffix_or = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix_or[i] = suffix_or[i+1] | nums[i]
        
        best = 0
        # try applying all k operations to nums[i]
        shift = 1 << k  # multiply by 2^k
        for i in range(n):
            # OR of all except i
            other_or = prefix_or[i] | suffix_or[i+1]
            # value of nums[i] after k doublings
            new_val = nums[i] * shift
            best = max(best, other_or | new_val)
        
        return best