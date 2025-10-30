from collections import defaultdict
from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [1] * n
        last_occurrence = defaultdict(list)
        
        for i in range(n):
            for j in last_occurrence[nums[i]]:
                if i - j - 1 <= k:
                    dp[i] = max(dp[i], dp[j] + 1)
            last_occurrence[nums[i]].append(i)
        
        return max(dp)