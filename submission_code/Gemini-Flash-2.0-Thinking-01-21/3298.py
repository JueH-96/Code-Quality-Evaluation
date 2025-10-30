from collections import defaultdict

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        dp = defaultdict(int)
        max_len = 0
        for num in nums:
            v1 = num
            v2 = num + 1
            l1 = dp[v1 - 1] + 1
            l2 = dp[v2 - 1] + 1
            dp[v1] = max(dp[v1], l1)
            dp[v2] = max(dp[v2], l2)
            max_len = max(max_len, dp[v1], dp[v2])
        return max_len