from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Compute the AND of the entire array
        total_and = ~0
        for x in nums:
            total_and &= x
        
        # If the total AND is positive, the minimum possible sum of ANDs
        # is total_and itself (by taking the whole array as one segment).
        # Any further split will only increase the sum.
        if total_and > 0:
            return 1
        
        # Otherwise, we can achieve a total sum of 0 by making each segment's AND = 0.
        # Greedily cut at the earliest point where the running AND becomes zero.
        count = 0
        curr_and = ~0
        for x in nums:
            curr_and &= x
            if curr_and == 0:
                # we found a segment ending here with AND = 0
                count += 1
                curr_and = ~0
        
        # If we never found any zero-AND segment (count == 0),
        # but total_and == 0, then the whole array as one segment gives AND = 0.
        return count if count > 0 else 1