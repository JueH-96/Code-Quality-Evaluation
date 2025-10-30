class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i][j] represents the length of longest subsequence ending at index i
        # with previous element at index j
        dp = [[1] * (n + 1) for _ in range(n)]
        max_length = 1
        
        # For each position i
        for i in range(1, n):
            # Compare with all previous positions j
            for j in range(i):
                # Get current absolute difference
                curr_diff = abs(nums[i] - nums[j])
                
                # For each previous position k before j
                for k in range(j):
                    # Get previous absolute difference
                    prev_diff = abs(nums[j] - nums[k])
                    
                    # If current difference is less than or equal to previous difference
                    if curr_diff <= prev_diff:
                        # Update dp[i][j] with maximum possible length
                        dp[i][j] = max(dp[i][j], dp[j][k] + 1)
                
                # Handle case when j is the first element in subsequence
                dp[i][j] = max(dp[i][j], 2)
                max_length = max(max_length, dp[i][j])
        
        return max_length