from functools import lru_cache
from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        @lru_cache(None)
        def dfs(index, prev, count):
            if index == len(nums):
                return 0
            # Skip current element
            max_len = dfs(index + 1, prev, count)
            if nums[index] != prev and count < k:
                # Include current element and potentially increment count
                max_len = max(max_len, 1 + dfs(index + 1, nums[index], count + 1))
            else:
                # Include current element when it's the same as previous
                max_len = max(max_len, dfs(index + 1, nums[index], count))
            return max_len
        
        return dfs(0, None, 0) + 1