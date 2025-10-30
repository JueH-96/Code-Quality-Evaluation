from typing import List
import bisect
from functools import lru_cache

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # Compute all unique differences
        unique_diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                unique_diffs.add(nums[j] - nums[i])
        unique_diffs = sorted(unique_diffs)
        
        # DP function to count subsequences with min difference at least d
        @lru_cache(maxsize=None)
        def count(start, k, d):
            if k == 0:
                return 1
            if start == n:
                return 0
            # Option 1: skip nums[start]
            option1 = count(start + 1, k, d)
            # Option 2: choose nums[start], find next index with nums[next] - nums[start] >= d
            next_index = bisect.bisect_left(nums, nums[start] + d, lo=start+1, hi=n)
            if next_index < n:
                option2 = count(next_index, k-1, d)
            else:
                option2 = 0
            return (option1 + option2) % MOD
        
        total_sum = 0
        # Iterate over each unique difference d
        for d in unique_diffs:
            count_at_least_d = count(0, k, d)
            # Find count_strictly_greater_than_d
            count_strictly_greater_than_d = count(0, k, d+1) if d+1 in unique_diffs else 0
            # count_d = count_at_least_d - count_strictly_greater_than_d
            count_d = (count_at_least_d - count_strictly_greater_than_d + MOD) % MOD
            # Add d * count_d to total_sum
            total_sum = (total_sum + d * count_d) % MOD
            # Clear the cache for the next iteration
            count.cache_clear()
        
        return total_sum