class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix sums for O(1) subarray sum access
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Calculate weights for each position
        weights = [0] * (k + 1)
        for j in range(1, k + 1):
            weights[j] = (1 if j % 2 == 1 else -1) * (k - j + 1)
        
        # Initialize dp array with negative infinity
        dp = [[float('-inf')] * (k + 1) for _ in range(n + 1)]
        
        # Base case: selecting 0 subarrays has a strength of 0
        for i in range(n + 1):
            dp[i][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                # Option 1: Don't include element i in any subarray
                dp[i][j] = dp[i-1][j]
                
                # Option 2: Include element i as the end of a new subarray
                for l in range(1, i + 1):
                    # Calculate sum of subarray from l to i
                    subarray_sum = prefix_sum[i] - prefix_sum[l - 1]
                    
                    # Add this subarray's contribution to the best arrangement of j-1 subarrays
                    # from the elements before position l
                    dp[i][j] = max(dp[i][j], dp[l-1][j-1] + subarray_sum * weights[j])
        
        return dp[n][k]