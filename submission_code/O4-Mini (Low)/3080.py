from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # We want to split into as many subarrays as possible each having AND == 0,
        # because that drives the total score to 0 (the global minimum) if possible.
        # Greedily accumulate the bitwise AND in `acc`. Whenever acc becomes 0,
        # we finalize one subarray, increment our count, and reset acc to all-ones
        # to start a new segment. If we never hit acc == 0, we cannot achieve total
        # score 0, the best we can do is one segment whose AND is the AND of the whole array.
        
        # Since nums[i] <= 10^6, we only need about 20 bits of 1's for the reset value:
        ALL_ONES = (1 << 20) - 1
        
        count = 0
        acc = ALL_ONES
        for x in nums:
            acc &= x
            if acc == 0:
                count += 1
                acc = ALL_ONES
        
        # If we have found at least one zero-AND segment, that's the global minimum sum = 0,
        # and we've maximized the number of such segments in `count`.
        # If count == 0, we never reached AND == 0, so the minimum total score
        # is the AND of the entire array, and we can only have one segment.
        return count if count > 0 else 1