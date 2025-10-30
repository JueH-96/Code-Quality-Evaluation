from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        # For each index i, we want to check if we can form exactly nums[i]
        # using a subset of query contributions that affect index i.
        # We'll maintain a DP for each index i representing the achievable sums.
        # We represent the DP with a bitmask integer: if the bit at position s is 1 it means sum s can be achieved.
        # Initially for every index, the only achievable sum is 0.
        dp = [1] * n
        # mask for each index to restrict bits up to nums[i] (we don't care about sums larger than nums[i], since overshooting is useless)
        masks = [(1 << (target + 1)) - 1 for target in nums]
        
        # Check if initial array is already a Zero Array.
        # Since dp[i] always has bit0 set and nums[i]==0 means target 0 is already achieved.
        if all(num == 0 for num in nums):
            return 0
        
        # Process queries one by one.
        for k, query in enumerate(queries, 1):
            l, r, val = query
            # For each index in the given range, update its dp.
            for i in range(l, r + 1):
                # We update: newdp = olddp OR (olddp shifted left by "val"), but we only care bits up to nums[i]
                dp[i] = (dp[i] | (dp[i] << val)) & masks[i]
            # After processing this query, check if for every index i, we can achieve sum exactly nums[i].
            if all((dp[i] >> nums[i]) & 1 for i in range(n)):
                return k
        return -1