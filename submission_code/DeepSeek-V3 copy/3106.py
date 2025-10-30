class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # Initialize a DP array where dp[i][j] represents the maximum length of subsequence
        # using the first i elements to sum up to j
        dp = [[-1] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(target + 1):
                # If we don't take the current number
                if dp[i - 1][j] != -1:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j])
                # If we take the current number
                if j >= num and dp[i - 1][j - num] != -1:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - num] + 1)
        
        return dp[n][target] if dp[n][target] != -1 else -1