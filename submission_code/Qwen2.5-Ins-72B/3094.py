from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        operations = 0
        
        for freq in count.values():
            if freq == 1:
                return -1
            operations += freq // 3
            if freq % 3 != 0:
                operations += 1
        
        return operations