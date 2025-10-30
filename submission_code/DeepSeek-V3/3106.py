class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # Initialize a DP array with -1, indicating that the sum is not achievable
        dp = [-1] * (target + 1)
        dp[0] = 0  # sum 0 can be achieved with 0 elements
        
        for num in nums:
            # Iterate from target down to num to avoid overwriting
            for t in range(target, num - 1, -1):
                if dp[t - num] != -1:
                    if dp[t] == -1:
                        dp[t] = dp[t - num] + 1
                    else:
                        dp[t] = max(dp[t], dp[t - num] + 1)
        
        return dp[target]