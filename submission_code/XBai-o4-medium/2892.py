from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_num = max(nums)
        n_length = len(nums)
        
        if n_length != max_num + 1:
            return False
        
        count = Counter(nums)
        
        # Check numbers from 1 to max_num - 1
        for num in range(1, max_num):
            if count.get(num, 0) != 1:
                return False
        
        # Check the max_num occurs exactly twice
        if count.get(max_num, 0) != 2:
            return False
        
        return True