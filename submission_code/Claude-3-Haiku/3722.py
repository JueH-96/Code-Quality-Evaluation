class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            max_sum = float('-inf')
            for j in range(1, min(i, m) + 1):
                max_sum = max(max_sum, sum(nums[i - j:i]))
            dp[i][1] = max_sum
            
            for j in range(2, min(i // m, k) + 1):
                max_sum = float('-inf')
                for l in range(m, i + 1):
                    max_sum = max(max_sum, dp[i - l][j - 1] + sum(nums[i - l:i]))
                dp[i][j] = max_sum
        
        return dp[n][k]