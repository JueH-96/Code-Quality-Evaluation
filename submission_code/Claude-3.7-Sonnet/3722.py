class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Create a 2D DP array
        # dp[i][j] = maximum sum of j non-overlapping subarrays, each with length at least m,
        # using the first i elements of nums
        dp = [[-float('inf')] * (k+1) for _ in range(n+1)]
        
        # Base case: 0 subarrays means sum of 0
        for i in range(n+1):
            dp[i][0] = 0
        
        # Precompute prefix sums for efficient subarray sum calculations
        prefix_sum = [0] * (n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        
        # Fill the DP array
        for i in range(1, n+1):  # Consider the first i elements
            for j in range(1, min(k+1, i//m + 1)):  # Consider j subarrays
                # Option 1: Don't include the current element in any subarray
                dp[i][j] = dp[i-1][j]
                
                # Option 2: Include the current element as the end of a subarray
                # Try all possible starting points for this subarray
                for l in range(max(0, i-m+1)):
                    # The subarray is [l...i-1] with length i-l
                    # We need i-l >= m, which is guaranteed by our loop bounds
                    subarray_sum = prefix_sum[i] - prefix_sum[l]
                    dp[i][j] = max(dp[i][j], dp[l][j-1] + subarray_sum)
        
        return dp[n][k]