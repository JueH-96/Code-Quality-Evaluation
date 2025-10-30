class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        
        # dp[i][s] = length of the longest subsequence using first i elements (0..i-1),
        #            whose sum is exactly s. If no subsequence can form sum s,
        #            store -1.
        # We'll build this table up to n rows and target+1 columns.
        dp = [[-1] * (target + 1) for _ in range(n + 1)]
        
        # Base case: sum of 0 can be formed with an empty subsequence of length 0
        for i in range(n + 1):
            dp[i][0] = 0
        
        for i in range(1, n + 1):
            for s in range(target + 1):
                # Option 1: don't take nums[i-1]
                dp[i][s] = dp[i-1][s]
                
                # Option 2: take nums[i-1] if possible
                if s >= nums[i-1] and dp[i-1][s - nums[i-1]] != -1:
                    dp[i][s] = max(dp[i][s], dp[i-1][s - nums[i-1]] + 1)
        
        return dp[n][target] if dp[n][target] != -1 else -1