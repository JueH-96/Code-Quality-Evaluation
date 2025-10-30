class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[s] will hold the maximum length of a subsequence with sum = s.
        # Initialize with -1 to denote "not possible", except dp[0] = 0 (empty subsequence).
        dp = [-1] * (target + 1)
        dp[0] = 0
        
        for num in nums:
            # Traverse in reverse to not overwrite previous states we still need to use
            for s in range(target - num, -1, -1):
                if dp[s] != -1:
                    new_sum = s + num
                    dp[new_sum] = max(dp[new_sum], dp[s] + 1)
        
        return dp[target]