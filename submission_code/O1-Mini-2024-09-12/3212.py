from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        dp = [0] * (n + 1)
        dp[0] = 1
        prefix_sum = [0] * (n + 1)
        prefix_sum[0] = 1
        last_occurrence = {}
        last_split = 0

        for i in range(1, n + 1):
            num = nums[i - 1]
            if num in last_occurrence:
                last_split = max(last_split, last_occurrence[num] + 1)
            last_occurrence[num] = i - 1
            dp[i] = (prefix_sum[i - 1] - (prefix_sum[last_split - 1] if last_split > 0 else 0)) % mod
            prefix_sum[i] = (prefix_sum[i - 1] + dp[i]) % mod

        return dp[n]