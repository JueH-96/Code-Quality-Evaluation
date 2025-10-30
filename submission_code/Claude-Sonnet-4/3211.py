class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # dp[i] represents max length ending at position i
        # last[i] represents the last value in optimal sequence ending at position i
        dp = [0] * (n + 1)
        last = [0] * (n + 1)
        
        for i in range(1, n + 1):
            dp[i] = 1
            last[i] = prefix[i]
            
            for j in range(i):
                segment_sum = prefix[i] - prefix[j]
                if last[j] <= segment_sum and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    last[i] = segment_sum
        
        return dp[n]