class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        # Calculate prefix sum for quick range sum queries
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # dp[i] = max length of non-decreasing array ending at index i
        # last[i] = the last value in the optimal solution ending at i
        dp = [0] * n
        last = [0] * n
        
        # Base case: One element array
        dp[0] = 1
        last[0] = nums[0]
        
        # For each position, try all possible previous positions
        for i in range(1, n):
            # Default: If we just merge all elements from 0 to i
            dp[i] = 1
            last[i] = prefix_sum[i + 1]
            
            for j in range(i):
                # Calculate sum of elements from j+1 to i
                segment_sum = prefix_sum[i + 1] - prefix_sum[j + 1]
                
                # If we can add this segment to the array ending at j
                # and it maintains the non-decreasing property
                if last[j] <= segment_sum and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    last[i] = segment_sum
        
        return dp[n - 1]