from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Set of values we still need to collect
        needed = set(range(1, k + 1))
        operations = 0
        
        # Traverse the array from the end (simulate popping)
        for num in reversed(nums):
            operations += 1          # One pop operation
            if num in needed:        # If this number is still needed, remove it
                needed.remove(num)
                if not needed:       # Collected everything we need
                    return operations
        
        # The problem guarantees all needed numbers exist, 
        # but we return operations for completeness.
        return operations