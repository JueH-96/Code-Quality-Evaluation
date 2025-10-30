class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[j] = length of longest subsequence that sums to j
        dp = [-1] * (target + 1)
        dp[0] = 0  # empty subsequence sums to 0
        
        for num in nums:
            # Iterate from target down to num to avoid using same element multiple times
            for j in range(target, num - 1, -1):
                if dp[j - num] != -1:  # If we can form sum (j - num)
                    dp[j] = max(dp[j], dp[j - num] + 1)
        
        return dp[target]