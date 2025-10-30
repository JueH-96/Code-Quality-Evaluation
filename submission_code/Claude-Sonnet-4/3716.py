class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        
        # dp[i][j] = length of longest subsequence ending at index i 
        # where j is the index of the second-to-last element
        dp = {}
        
        # Initialize: every pair of elements can form a subsequence of length 2
        for i in range(1, n):
            for j in range(i):
                dp[(i, j)] = 2
        
        max_length = 2
        
        # For each position k, try to extend existing subsequences
        for k in range(2, n):
            for i in range(1, k):
                for j in range(i):
                    # Current subsequence ends at i with second-to-last at j
                    # Try to extend it by adding nums[k]
                    
                    prev_diff = abs(nums[i] - nums[j])
                    curr_diff = abs(nums[k] - nums[i])
                    
                    # Check if we can extend (current diff <= previous diff)
                    if curr_diff <= prev_diff:
                        if (k, i) not in dp:
                            dp[(k, i)] = 0
                        dp[(k, i)] = max(dp[(k, i)], dp[(i, j)] + 1)
                        max_length = max(max_length, dp[(k, i)])
        
        return max_length