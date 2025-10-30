from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # We need to collect all numbers from 1 to k.
        needed = set(range(1, k + 1))
        seen = set()
        operations = 0
        
        # Start removing elements from the end until we've seen all required elements.
        for num in reversed(nums):
            operations += 1
            if num in needed:
                seen.add(num)
            if len(seen) == k:
                return operations
        
        # Given the constraints (collection is always possible),
        # the loop will always return before reaching here.
        return operations