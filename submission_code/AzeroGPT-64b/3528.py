class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        # Standard DP problem.
        dp = [0] * len(nums)
        dp[0] = 0
        max_score = 0
        # index of the longest valid prefix
        valid_prefix_end = 0
        
        for i in range(1, len(nums)):
            # make sure nums[i] > nums[idx] of (idx < i) for valid_prefix_end
            while valid_prefix_end < i and nums[valid_prefix_end + 1] > nums[valid_prefix_end]:
                valid_prefix_end += 1
            
            dp[i] = max(dp[j] + (i - j) * nums[j] for j in range(0, valid_prefix_end + 1))
            max_score = max(max_score, dp[i])
        
        return max_score