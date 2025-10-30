class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i] will store the number of good partitions of the subarray nums[0:i]
        dp = [0] * (n + 1)
        dp[0] = 1  # There's one way to partition an empty array
        
        last_occurrence = {}
        
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            if nums[i - 1] in last_occurrence:
                dp[i] = (dp[i] - dp[last_occurrence[nums[i - 1]] - 1]) % MOD
            last_occurrence[nums[i - 1]] = i
        
        return dp[n]