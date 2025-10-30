class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Compute prefix sums for efficient subarray sum calculation
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        # dp[i][j] = max sum using i subarrays from first j elements
        # Initialize with negative infinity
        dp = [[float('-inf')] * (n + 1) for _ in range(k + 1)]
        
        # Base case: 0 subarrays selected = 0 sum
        for j in range(n + 1):
            dp[0][j] = 0
        
        # Fill the DP table
        for i in range(1, k + 1):
            # Need at least i*m elements to form i subarrays of length >= m
            for j in range(i * m, n + 1):
                # Option 1: Don't use element at index j-1
                dp[i][j] = dp[i][j-1]
                
                # Option 2: End a subarray at index j-1
                # Try all possible starting positions for this subarray
                for start in range(j - m + 1):
                    # Subarray from start to j-1 (inclusive)
                    subarray_sum = prefix[j] - prefix[start]
                    # Add this to the best solution for i-1 subarrays ending before start
                    if start >= (i-1) * m:  # Ensure we have room for i-1 subarrays
                        dp[i][j] = max(dp[i][j], dp[i-1][start] + subarray_sum)
        
        return dp[k][n]