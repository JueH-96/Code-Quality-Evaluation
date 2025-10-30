from typing import List
from collections import defaultdict

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if total < target:
            return -1
        if total == target:
            return 0
        
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        splits = 0
        
        for power in range(30, -1, -1):
            bit = 1 << power
            if (target & bit) == 0:
                continue
            
            while freq[bit] == 0:
                found = False
                for p in range(power - 1, -1, -1):
                    parent = 1 << p
                    if freq[parent] > 0:
                        splits += 1
                        freq[parent] -= 1
                        freq[parent >> 1] += 2
                        found = True
                        break
                if not found:
                    return -1
        
        return splits