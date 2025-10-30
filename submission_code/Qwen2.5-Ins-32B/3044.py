from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        operations = 0
        
        while len(collected) < k:
            element = nums.pop()
            operations += 1
            if 1 <= element <= k:
                collected.add(element)
        
        return operations