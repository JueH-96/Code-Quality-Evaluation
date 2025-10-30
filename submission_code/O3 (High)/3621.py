from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # It is impossible to increase any element.
        # Hence if some element is already smaller than k we can never reach k.
        if any(x < k for x in nums):
            return -1
        
        # Each operation can eliminate at most one distinct value
        # that is larger than k (see analysis).
        # Therefore the minimal number of operations equals the number
        # of distinct values that are strictly greater than k.
        return len({x for x in nums if x > k})