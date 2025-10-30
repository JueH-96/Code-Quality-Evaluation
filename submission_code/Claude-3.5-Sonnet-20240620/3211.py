class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        last = [0] * (n + 1)
        sum = [0] * (n + 1)
        
        for i in range(1, n + 1):
            sum[i] = sum[i-1] + nums[i-1]
            j = last[i-1]
            while j < i and sum[i] - sum[j] < sum[j] - sum[last[j]]:
                j += 1
            dp[i] = dp[j] + 1
            last[i] = j
        
        return dp[n]