class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        
        # dp[i][diff] = length of longest subsequence ending at i with last difference = diff
        dp = [[0] * 300 for _ in range(n)]
        max_length = 2
        
        for i in range(1, n):
            for j in range(i):
                abs_diff = abs(nums[i] - nums[j])
                
                # Start with a subsequence of length 2
                current_length = 2
                
                # Try to extend from previous subsequences ending at j
                for prev_diff in range(abs_diff, 300):
                    if dp[j][prev_diff] > 0:
                        current_length = max(current_length, dp[j][prev_diff] + 1)
                
                # Update dp and max_length
                dp[i][abs_diff] = max(dp[i][abs_diff], current_length)
                max_length = max(max_length, dp[i][abs_diff])
        
        return max_length