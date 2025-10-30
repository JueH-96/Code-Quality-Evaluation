from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Use a large sentinel for initialization
        INF = 10**30
        
        # min_prefix[m] will store the minimum prefix sum seen so far
        # at an index whose parity mod k is m.
        min_prefix = [INF] * k
        # prefix sum up to index 0 is 0, and 0 % k = 0
        min_prefix[0] = 0
        
        curr_sum = 0
        max_sum = -INF
        
        # We iterate r from 1 to n, computing prefix sum up to r.
        # The subarray ending at r-1 and starting at l has length
        # divisible by k if r % k == l % k.
        for r, v in enumerate(nums, start=1):
            curr_sum += v
            m = r % k
            # If we've seen a prefix with the same mod class,
            # consider forming a subarray from that prefix to here.
            if min_prefix[m] != INF:
                candidate = curr_sum - min_prefix[m]
                if candidate > max_sum:
                    max_sum = candidate
            # Update the minimum prefix sum for this mod class
            if curr_sum < min_prefix[m]:
                min_prefix[m] = curr_sum
        
        return max_sum