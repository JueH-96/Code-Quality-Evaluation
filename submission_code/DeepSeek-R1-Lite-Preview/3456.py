from typing import List
from collections import defaultdict

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        dp = [defaultdict(int) for _ in range(k + 1)]
        
        for num in nums:
            for j in range(k + 1):
                for last_val, length in list(dp[j].items()):
                    if num == last_val:
                        dp[j][num] = max(dp[j][num], length + 1)
                    else:
                        if j < k:
                            dp[j + 1][num] = max(dp[j + 1][num], length + 1)
                dp[j][num] = max(dp[j][num], 1)
        
        max_length = 0
        for j in range(k + 1):
            if dp[j]:
                max_length = max(max_length, max(dp[j].values()))
        
        return max_length