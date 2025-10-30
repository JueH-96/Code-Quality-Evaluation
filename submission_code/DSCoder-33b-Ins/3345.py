class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        dp = [[0]*(n+1) for _ in range(n+1)]
        pre_sum = [0]*(n+1)
        for i in range(n-1, -1, -1):
            dp[i][1] = nums[i]
            pre_sum[i] = (pre_sum[i+1] + dp[i][1]) % MOD
            for j in range(2, n-i+1):
                dp[i][j] = (dp[i+1][j-1] + nums[i]) % MOD
                pre_sum[i] = (pre_sum[i] + dp[i][j]) % MOD
        ans = 0
        for i in range(n):
            if nums[i] <= k:
                ans = (ans + pre_sum[i+1]) % MOD
        return ans