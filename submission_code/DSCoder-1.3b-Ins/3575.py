class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = max(prefix[i], nums[i])
        for i in range(n-1, -1, -1):
            suffix[i] = max(suffix[i+1], nums[i])
        dp = [0]*(n+1)
        for i in range(n):
            dp[i+1] = max(dp[i], prefix[i]+suffix[i+1])
        ans = 0
        for i in range(k):
            ans = max(ans, dp[i]^dp[n-k+i])
        return ans