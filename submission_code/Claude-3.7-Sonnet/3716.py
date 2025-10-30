class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        
        # dp[(i, j)] represents the length of the longest subsequence
        # ending with elements at indices i and j
        dp = {}
        max_length = 2  # Since n >= 2, the longest subsequence is at least 2
        
        for i in range(n):
            for j in range(i + 1, n):
                dp[(i, j)] = 2  # Base case: just the pair (i, j)
                diff = abs(nums[j] - nums[i])
                
                for k in range(i):
                    prev_diff = abs(nums[i] - nums[k])
                    if prev_diff >= diff:
                        dp[(i, j)] = max(dp[(i, j)], dp[(k, i)] + 1)
                
                max_length = max(max_length, dp[(i, j)])
        
        return max_length