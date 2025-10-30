class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][j] represents max strength using elements 0..i-1 with exactly j subarrays
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        
        # Base case: 0 subarrays gives 0 strength
        for i in range(n + 1):
            dp[i][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                # Case 1: Don't include nums[i-1] in any subarray
                dp[i][j] = dp[i-1][j]
                
                # Case 2: Include nums[i-1] as the end of the j-th subarray
                coeff = (1 if j % 2 == 1 else -1) * (k - j + 1)
                subarray_sum = 0
                
                # Try all possible starts for this subarray ending at i-1
                for start in range(i-1, -1, -1):
                    subarray_sum += nums[start]
                    if dp[start][j-1] != -float('inf'):
                        dp[i][j] = max(dp[i][j], dp[start][j-1] + coeff * subarray_sum)
        
        return dp[n][k]