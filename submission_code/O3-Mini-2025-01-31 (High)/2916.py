from typing import List
from functools import lru_cache

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # Build prefix sum array for quick range sum queries.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Helper function: get sum of nums[i:j+1]
        def range_sum(i: int, j: int) -> int:
            return prefix_sum[j + 1] - prefix_sum[i]
        
        # Use memoization for intervals [i, j]
        @lru_cache(maxsize=None)
        def can_split(i: int, j: int) -> bool:
            # Base case: single element segment is already "split"
            if i == j:
                return True
            
            # Try all possible splits: split at index k gives left: nums[i:k+1], right: nums[k+1:j+1]
            for k in range(i, j):
                left_len = k - i + 1
                right_len = j - k
                left_valid = (left_len == 1) or (range_sum(i, k) >= m)
                right_valid = (right_len == 1) or (range_sum(k + 1, j) >= m)
                # If both parts pass the current split condition, then they must be further splittable.
                if left_valid and right_valid:
                    if can_split(i, k) and can_split(k + 1, j):
                        return True
            return False

        return can_split(0, n - 1)