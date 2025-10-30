class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        
        # dp[i][j] = length of longest valid subsequence ending at index i 
        # with the previous element at index j (j < i)
        dp = [[0] * n for _ in range(n)]
        
        # Base case: any two elements form a valid subsequence of length 2
        for i in range(1, n):
            for j in range(i):
                dp[i][j] = 2
        
        max_length = 2
        
        # DP: try to extend existing subsequences
        for i in range(2, n):  # current position
            for j in range(1, i):  # previous position in the subsequence
                for k in range(j):  # position before the previous one
                    # Check if we can extend the subsequence ending at j with prev at k
                    prev_diff = abs(nums[j] - nums[k])
                    curr_diff = abs(nums[i] - nums[j])
                    if curr_diff <= prev_diff:
                        dp[i][j] = max(dp[i][j], dp[j][k] + 1)
                        max_length = max(max_length, dp[i][j])
        
        return max_length