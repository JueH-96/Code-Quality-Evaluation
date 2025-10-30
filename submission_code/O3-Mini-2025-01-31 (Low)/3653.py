from typing import List
import math

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        # We'll use prefix sum and a dictionary to store the minimum prefixSum encountered with a given remainder mod k.
        best = {}
        best[0] = 0  # prefix sum 0 at index -1, remainder 0
        prefix = 0
        res = -math.inf
        
        for num in nums:
            prefix += num
            mod = prefix % k  # non-negative remainder due to modulo definition in Python
            if mod in best:
                # The subarray sum ending at the current index with start index where remainder is same is prefix - best[mod]
                res = max(res, prefix - best[mod])
            # Update the minimum prefix sum for current remainder for future subarrays.
            if mod in best:
                best[mod] = min(best[mod], prefix)
            else:
                best[mod] = prefix
        
        return res