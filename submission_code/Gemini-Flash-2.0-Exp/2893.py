class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [[float('-inf')] * 2 for _ in range(n)]
        
        # dp[i][0] stores the maximum score ending at index i with nums[i] being even
        # dp[i][1] stores the maximum score ending at index i with nums[i] being odd
        
        dp[0][nums[0] % 2] = nums[0]
        
        for i in range(1, n):
            for j in range(2):
                dp[i][nums[i] % 2] = nums[i]
                for k in range(i):
                    parity_i = nums[i] % 2
                    parity_k = nums[k] % 2
                    
                    cost = 0 if parity_i == parity_k else x
                    
                    dp[i][parity_i] = max(dp[i][parity_i], dp[k][parity_k] + nums[i] - cost)
        
        return max(dp[n-1][0], dp[n-1][1])