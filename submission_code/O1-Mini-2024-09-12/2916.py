from typing import List
from functools import lru_cache

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        @lru_cache(None)
        def can_split(l, r):
            if l == r:
                return True
            for k in range(l, r):
                sum_left = prefix[k + 1] - prefix[l]
                sum_right = prefix[r + 1] - prefix[k + 1]
                len_left = k - l + 1
                len_right = r - k

                valid_left = (len_left == 1) or (sum_left >= m)
                valid_right = (len_right == 1) or (sum_right >= m)

                if valid_left and valid_right:
                    if can_split(l, k) and can_split(k + 1, r):
                        return True
            return False

        return can_split(0, n - 1)