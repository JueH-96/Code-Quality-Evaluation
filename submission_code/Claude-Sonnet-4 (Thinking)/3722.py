class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Precompute prefix sums for efficient subarray sum calculation
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # dp[i][j] = maximum sum using exactly j subarrays from nums[0:i]
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        
        # Base case: using 0 subarrays gives sum 0
        for i in range(n + 1):
            dp[i][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                # Option 1: don't include nums[i-1] in any subarray
                dp[i][j] = dp[i-1][j]
                
                # Option 2: end a subarray at position i-1
                # Try all possible lengths from m to i
                for length in range(m, i + 1):
                    start_pos = i - length
                    subarray_sum = prefix[i] - prefix[start_pos]
                    if dp[start_pos][j-1] != -float('inf'):
                        dp[i][j] = max(dp[i][j], dp[start_pos][j-1] + subarray_sum)
        
        return dp[n][k]