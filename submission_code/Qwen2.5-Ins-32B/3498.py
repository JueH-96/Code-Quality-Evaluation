from typing import List
from collections import defaultdict
from functools import lru_cache

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        group_counts = defaultdict(lambda: defaultdict(int))
        min_changes = [n] * (k + 1)
        
        for i in range(n // 2):
            a, b = nums[i], nums[n - i - 1]
            for x in range(k + 1):
                group_counts[i % (n // 2)][x] += (abs(a - x) != x and abs(b - x) != x)
        
        for i in range(n // 2):
            for x in range(k + 1):
                min_changes[x] = min(min_changes[x], group_counts[i % (n // 2)][x])
        
        @lru_cache(None)
        def dp(index, prev_x):
            if index == n // 2:
                return 0
            min_cost = float('inf')
            for x in range(k + 1):
                cost = group_counts[index % (n // 2)][x] + (abs(x - prev_x) != 0)
                min_cost = min(min_cost, cost + dp(index + 1, x))
            return min_cost
        
        return min(dp(0, x) for x in range(k + 1))