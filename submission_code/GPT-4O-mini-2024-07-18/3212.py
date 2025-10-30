class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        last_seen = {}
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to partition an empty array
        
        for i in range(1, n + 1):
            num = nums[i - 1]
            dp[i] = dp[i - 1] * 2 % MOD  # Double the ways from the previous index
            
            if num in last_seen:
                j = last_seen[num]
                dp[i] = (dp[i] - dp[j - 1] + MOD) % MOD  # Remove invalid partitions
            
            last_seen[num] = i  # Update the last seen index of the current number
        
        return dp[n]