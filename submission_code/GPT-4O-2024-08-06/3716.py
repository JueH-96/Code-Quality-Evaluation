class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        # dp[i] will store the length of the longest subsequence ending at index i
        dp = [1] * n
        
        # Iterate over each pair of indices i, j with i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate the absolute difference
                diff_ij = abs(nums[j] - nums[i])
                
                # Check if we can extend the subsequence ending at i
                # by including nums[j]
                for k in range(i):
                    diff_ki = abs(nums[i] - nums[k])
                    if diff_ki >= diff_ij:
                        dp[j] = max(dp[j], dp[i] + 1)
        
        # The answer is the maximum value in dp array
        return max(dp)