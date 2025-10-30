class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0]*n
        dp[0] = 1
        for i in range(1, n):
            dp[i] = dp[i-1]
            if i >= 2:
                dp[i] = (dp[i]*2)%MOD
            if i >= 3:
                dp[i] = (dp[i] + dp[i-3])%MOD
            if nums[i] > nums[i-1]:
                dp[i] = (dp[i]*2)%MOD
                if i >= 2 and nums[i] > nums[i-2]:
                    dp[i] = (dp[i] + 2)%MOD
        return sum(dp)%MOD