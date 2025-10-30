class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [[-1] * (target + 1) for _ in range(len(nums) + 1)]
        dp[0][0] = 0
        
        for i in range(1, len(nums) + 1):
            dp[i][0] = 0
            for j in range(target + 1):
                # Don't take current number
                dp[i][j] = dp[i-1][j]
                
                # Take current number if possible
                if j >= nums[i-1] and dp[i-1][j-nums[i-1]] != -1:
                    dp[i][j] = max(dp[i][j], dp[i-1][j-nums[i-1]] + 1)
                    
        return dp[len(nums)][target]