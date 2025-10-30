from typing import List
import math
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        n = len(nums)
        
        def is_possible(k):
            s = (n + k -1) // k  # ceiling division
            total = 0
            for freq in count.values():
                total += (freq + s -1) // s
                if total > k:
                    return False
            return True
        
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            if is_possible(mid):
                right = mid
            else:
                left = mid + 1
        return left