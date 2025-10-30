from typing import List
from functools import lru_cache

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        @lru_cache(maxsize=None)
        def can_split(i: int, j: int) -> bool:
            if i == j:
                return True
            for k in range(i, j):
                left_sum = prefix[k + 1] - prefix[i]
                left_len = k - i + 1
                left_ok = left_len == 1 or left_sum >= m
                right_sum = prefix[j + 1] - prefix[k + 1]
                right_len = j - k
                right_ok = right_len == 1 or right_sum >= m
                if left_ok and right_ok:
                    if can_split(i, k) and can_split(k + 1, j):
                        return True
            return False
        
        return can_split(0, n - 1)