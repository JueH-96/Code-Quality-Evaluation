from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        keys = sorted(count.keys())
        dp = [0] * len(keys)
        dp2 = [0] * len(keys)
        dp[0] = count[keys[0]]
        for i in range(1, len(keys)):
            if keys[i] == keys[i-1] + 1:
                dp[i] = max(dp[i-1], dp2[i-1] + count[keys[i]])
                dp2[i] = dp[i-1]
            else:
                dp[i] = max(dp[i-1], dp2[i-1])
                dp2[i] = dp[i-1] - count[keys[i-1]] + count[keys[i]]
        return len(nums) - max(dp[-1], dp2[-1])