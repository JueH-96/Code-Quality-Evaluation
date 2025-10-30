class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # Precompute prefix sums for quick subarray sum calculation
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        
        # Initialize DP table
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True  # Single element subarrays are always valid
        
        # Fill the DP table for subarrays of length >= 2
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                # Check all possible split points k
                for k in range(i, j):
                    sub1_sum = prefix_sum[k+1] - prefix_sum[i]
                    sub2_sum = prefix_sum[j+1] - prefix_sum[k+1]
                    # Check conditions for both subarrays
                    cond1 = (k - i + 1 == 1) or (sub1_sum >= m)
                    cond2 = (j - k == 1) or (sub2_sum >= m)
                    if cond1 and cond2 and dp[i][k] and dp[k+1][j]:
                        dp[i][j] = True
                        break  # No need to check further splits
        
        return dp[0][n-1]