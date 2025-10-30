class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        from collections import Counter
        n = len(nums)
        dp = [Counter() for _ in range(n+1)]
        for i in range(n-1, -1, -1):
            for j in range(i+2, n+1):
                if len(set(nums[i:j])) == j-i and len(set(nums[i:j])) == len(nums[i:j]) - k:
                    dp[i] = max(dp[i], dp[j], key = len)
                else:
                    dp[i] = max(dp[i], dp[j])
        return max(dp[0].values())