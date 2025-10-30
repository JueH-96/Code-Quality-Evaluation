class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        from collections import defaultdict
        import bisect
        
        # dp[d] will store the maximum length of a valid subsequence where the last difference is d
        dp = defaultdict(int)
        
        # Initialize the maximum length of subsequence
        max_length = 1
        
        # Iterate over each pair of elements in nums
        for i in range(len(nums)):
            for j in range(i):
                diff = abs(nums[i] - nums[j])
                
                # We can extend the subsequence ending with difference 'diff' or start a new one with length 2
                new_length = dp[(j, diff)] + 1 if (j, diff) in dp else 2
                
                # Update the dp for the current pair (i, diff)
                dp[(i, diff)] = max(dp[(i, diff)], new_length)
                
                # Update the global maximum length
                max_length = max(max_length, dp[(i, diff)])
        
        return max_length