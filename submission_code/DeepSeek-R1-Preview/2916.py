from typing import List
from functools import lru_cache

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        @lru_cache(maxsize=None)
        def can_split(start: int, end: int) -> bool:
            if start >= end:
                return True
            length = end - start + 1
            if length == 2:
                return True
            
            for k in range(start, end):
                sum_left = prefix[k + 1] - prefix[start]
                len_left = k - start + 1
                sum_right = prefix[end + 1] - prefix[k + 1]
                len_right = end - k
                
                left_ok = (sum_left >= m) or (len_left == 1)
                right_ok = (sum_right >= m) or (len_right == 1)
                
                if left_ok and right_ok:
                    if can_split(start, k) and can_split(k + 1, end):
                        return True
            
            return False
        
        return can_split(0, n - 1)