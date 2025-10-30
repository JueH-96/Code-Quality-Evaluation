class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for j in range(1, k + 1):
            max_temp = -float('inf')
            for i in range(1, n + 1):
                if i - m >= 0:
                    s = i - m + 1  # s ranges up to i-m+1 (1-based)
                    # s-1 is i-m
                    # The term for s-1 is dp[s-1][j-1] - prefix[s-1]
                    current_val = dp[i - m][j-1] - prefix[i - m]
                    if current_val > max_temp:
                        max_temp = current_val
                if max_temp != -float('inf'):
                    current_candidate = max_temp + prefix[i]
                    dp[i][j] = max(dp[i-1][j], current_candidate)
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n][k]