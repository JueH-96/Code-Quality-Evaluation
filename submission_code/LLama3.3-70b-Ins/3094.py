from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        operations = 0
        
        for num in count:
            if count[num] % 2 != 0:
                return -1
            operations += count[num] // 3
            count[num] %= 3
            operations += count[num] // 2
        
        return operations