class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        # dp[i][j] represents min sum for first i elements divided into j subarrays
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        # For each position i
        for i in range(1, n + 1):
            # For each number of subarrays j
            for j in range(1, min(i + 1, m + 1)):
                # Try all possible previous positions k
                curr_and = nums[i - 1]
                for k in range(i - 1, -1, -1):
                    if k > 0:
                        curr_and &= nums[k - 1]
                    # If current subarray AND equals required AND value
                    if curr_and == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k][j - 1] + nums[i - 1])
        
        return dp[n][m] if dp[n][m] != float('inf') else -1