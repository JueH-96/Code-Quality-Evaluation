from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        
        count = Counter()
        for num in nums:
            count[num] += 1
        
        operations = 0
        current = 1
        
        while current <= target:
            if current & target:
                if count[current] > 0:
                    count[current] -= 1
                else:
                    while current <= target and count[current] == 0:
                        operations += 1
                        current *= 2
                    count[current] -= 1
                    continue
            
            if count[current] > 1:
                count[current * 2] += count[current] // 2
            current *= 2
        
        return operations