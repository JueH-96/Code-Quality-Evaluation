class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i][j] = length of longest subsequence ending at index i 
        # where the last absolute difference is j
        dp = {}
        
        max_length = 1
        
        for i in range(n):
            # For each position i, consider all previous positions j
            for j in range(i):
                diff = abs(nums[i] - nums[j])
                
                # Case 1: Start a new subsequence of length 2
                if (i, diff) not in dp:
                    dp[(i, diff)] = 2
                else:
                    dp[(i, diff)] = max(dp[(i, diff)], 2)
                
                # Case 2: Extend existing subsequences ending at j
                for prev_diff in range(diff, 300):
                    if (j, prev_diff) in dp:
                        if (i, diff) not in dp:
                            dp[(i, diff)] = dp[(j, prev_diff)] + 1
                        else:
                            dp[(i, diff)] = max(dp[(i, diff)], dp[(j, prev_diff)] + 1)
                
                if (i, diff) in dp:
                    max_length = max(max_length, dp[(i, diff)])
        
        return max_length