class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        
        # Initialize DP array
        # dp[i][j] = length of longest subsequence using elements up to index i-1 that sums to j
        dp = [[0] + [-1] * target for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(target + 1):
                # Exclude the current element
                dp[i][j] = dp[i-1][j]
                
                # Include the current element if possible
                if j >= nums[i-1] and dp[i-1][j - nums[i-1]] != -1:
                    dp[i][j] = max(dp[i][j], dp[i-1][j - nums[i-1]] + 1)
        
        return dp[n][target]