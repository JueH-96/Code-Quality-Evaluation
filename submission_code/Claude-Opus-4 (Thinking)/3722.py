class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Compute prefix sums for efficient subarray sum calculation
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        # dp[i][j] = maximum sum of j non-overlapping subarrays from first i elements
        # Initialize with negative infinity
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        
        # Base case: 0 subarrays from any number of elements gives sum 0
        for i in range(n + 1):
            dp[i][0] = 0
        
        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                # Check if it's possible to have j subarrays from i elements
                if i < j * m:
                    continue
                
                # Option 1: don't include nums[i-1] in any subarray
                if i - 1 >= j * m:
                    dp[i][j] = dp[i-1][j]
                
                # Option 2: include nums[i-1] as the end of a subarray
                max_length = i - (j - 1) * m
                for length in range(m, max_length + 1):
                    # Subarray from i-length to i-1 (0-indexed)
                    subarray_sum = prefix[i] - prefix[i-length]
                    dp[i][j] = max(dp[i][j], dp[i-length][j-1] + subarray_sum)
        
        return dp[n][k]