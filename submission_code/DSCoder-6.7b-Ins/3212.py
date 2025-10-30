class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        dp = [0, 1]
        curr_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                dp[1] = dp[1] * 2 % MOD
            else:
                dp = [dp[1], (dp[0] + dp[1]) % MOD]
            curr_sum += nums[i]
        if curr_sum - nums[0] == nums[0]:
            return dp[1] * 2 % MOD
        return dp[1]