class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i] represents the maximum score we can get starting from index i
        dp = [0] * n
        
        # Base case: dp[n-1] = 0 (already at the last index)
        
        # Fill dp array from n-2 down to 0
        for i in range(n-2, -1, -1):
            max_score = 0
            for j in range(i+1, n):
                # Calculate score for jumping from i to j
                jump_score = (j - i) * nums[i]
                # Add the maximum score we can get starting from j
                total_score = jump_score + dp[j]
                max_score = max(max_score, total_score)
            dp[i] = max_score
        
        return dp[0]