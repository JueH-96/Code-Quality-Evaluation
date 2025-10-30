from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        needed = set(range(1, k + 1))
        collected = set()
        operations = 0
        
        # Remove elements from the end one by one
        for x in reversed(nums):
            operations += 1
            if x in needed:
                collected.add(x)
                if len(collected) == k:
                    break
        
        return operations