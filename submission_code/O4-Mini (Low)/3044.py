from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # We need to collect all elements from 1 through k
        needed = set(range(1, k+1))
        ops = 0
        
        # Remove elements from the end one by one
        for x in reversed(nums):
            ops += 1
            if x in needed:
                needed.remove(x)
                if not needed:
                    return ops
        
        # Problem statement guarantees it's always possible
        return ops