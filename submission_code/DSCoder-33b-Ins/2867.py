class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0]*(n+1)
        dp[0] = 1
        last_one = -1
        for i in range(1, n+1):
            if nums[i-1] == 1:
                if last_one == -1:
                    dp[i] = dp[i-1]
                else:
                    dp[i] = (dp[i-1] + dp[last_one]) % MOD
                last_one = i - 1
            else:
                dp[i] = dp[i-1]
        return dp[n]