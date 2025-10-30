from typing import List
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        groups = defaultdict(int)
        
        for x in nums:
            s = x
            i = 2
            while i * i <= s:
                while s % (i * i) == 0:
                    s //= (i * i)
                i += 1
            groups[s] += x
        
        max_sum = max(groups.values())
        return max_sum