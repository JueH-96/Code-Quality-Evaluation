from collections import Counter
from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [Counter() for _ in range(k)]
        freq = [Counter() for _ in range(k)]
        min_counts = [0] * k

        for i, num in enumerate(nums):
            dp[i % k][num] += 1

        for i in range(k):
            min_counts[i] = min(dp[i].values())
            for j in dp[i]:
                freq[i][j % (k // gcd(i, k))] += dp[i][j]

        dp = [Counter() for _ in range(k)]
        dp[0] = freq[0]
        for i in range(1, k):
            for j in freq[i]:
                for prev in dp[i - 1]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][prev] + freq[i][j] if (j - prev) % (k // gcd(i, k)) == 0 else 0)
            min_counts[i] = max(min_counts[i], max(dp[i].values()))

        return n - max(min_counts)