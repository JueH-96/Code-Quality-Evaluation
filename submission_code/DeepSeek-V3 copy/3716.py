class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # dp[i][j] represents the length of the longest subsequence ending at index i with the last difference being j
        # Since the differences can be up to 300 (since 1 <= nums[i] <= 300), we can limit j to 300
        # Initialize dp as a dictionary of dictionaries
        dp = {}
        for i in range(n):
            dp[i] = {}
        
        max_length = 1
        for i in range(n):
            for j in range(i):
                diff = abs(nums[i] - nums[j])
                # Initialize the current difference in dp[i] if not present
                if diff not in dp[i]:
                    dp[i][diff] = 1
                # Iterate through all possible previous differences
                for prev_diff in dp[j]:
                    if prev_diff >= diff:
                        if diff not in dp[i]:
                            dp[i][diff] = dp[j][prev_diff] + 1
                        else:
                            dp[i][diff] = max(dp[i][diff], dp[j][prev_diff] + 1)
            # Update the maximum length
            for diff in dp[i]:
                max_length = max(max_length, dp[i][diff])
        
        return max_length