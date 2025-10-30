class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(n):
            dp[i + 1][0] = 0
            for j in range(k):
                if dp[i][j] == -float('inf'):
                    continue
                    
                curr_sum = 0
                for l in range(i, n):
                    curr_sum += nums[l]
                    if j + 1 <= k:
                        multiplier = k - j if (j + 1) % 2 == 1 else -(k - j)
                        dp[l + 1][j + 1] = max(dp[l + 1][j + 1], 
                                             dp[i][j] + curr_sum * multiplier)
        
        return dp[n][k]