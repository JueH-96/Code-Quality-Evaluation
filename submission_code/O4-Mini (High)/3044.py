from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # We need to pop from the end until we've collected all values 1..k.
        required = set(range(1, k + 1))
        ops = 0
        
        # Traverse from the end, counting each pop.
        for x in reversed(nums):
            ops += 1
            if x in required:
                required.remove(x)
                # Once we've collected all, return the ops count.
                if not required:
                    return ops
        
        # Problem guarantees you can collect all 1..k, but just in case:
        return ops