class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # dp[i][j] will store the XOR score of subarray nums[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Initialize base case - single elements
        for i in range(n):
            dp[i][i] = nums[i]
        
        # Fill the DP table for XOR scores
        for length in range(2, n + 1):  # length of subarray
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = dp[i][j-1] ^ dp[i+1][j]
        
        # max_dp[i][j] will store the maximum XOR score in range [i, j]
        max_dp = [[0] * n for _ in range(n)]
        
        # Initialize base case
        for i in range(n):
            max_dp[i][i] = dp[i][i]
        
        # Fill the max_dp table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                max_dp[i][j] = max(dp[i][j], max_dp[i][j-1], max_dp[i+1][j])
        
        # Answer queries
        result = []
        for l, r in queries:
            result.append(max_dp[l][r])
        
        return result