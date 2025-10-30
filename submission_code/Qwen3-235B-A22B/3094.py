from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        total_operations = 0
        
        for freq in count.values():
            if freq == 1:
                return -1
            if freq % 3 == 0:
                total_operations += freq // 3
            elif freq % 3 == 1:
                # freq >= 4 as freq >=2 and mod 1, so subtract 4 to get two operations of 2
                # and then divide the remaining by 3
                total_operations += (freq - 4) // 3 + 2
            else:  # freq % 3 == 2
                # add one operation of 2 and then divide the remaining by 3
                total_operations += (freq // 3) + 1
        
        return total_operations