class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        last_seen = {}
        dp = [1] * (len(nums) + 1)

        for i in range(1, len(nums) + 1):
            dp[i] = (2 * dp[i - 1]) % MOD
            if nums[i - 1] in last_seen:
                dp[i] = (dp[i] - dp[last_seen[nums[i - 1]] - 1] + MOD) % MOD
            last_seen[nums[i - 1]] = i

        return dp[len(nums)] - 1