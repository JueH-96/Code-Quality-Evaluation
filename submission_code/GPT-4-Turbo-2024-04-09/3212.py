class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0] * (n + 1)
        dp[0] = 1
        last_seen = {}
        
        for i in range(n):
            dp[i + 1] = (2 * dp[i]) % MOD
            if nums[i] in last_seen:
                dp[i + 1] -= dp[last_seen[nums[i]]]
                dp[i + 1] %= MOD
            last_seen[nums[i]] = i
        
        return dp[n]