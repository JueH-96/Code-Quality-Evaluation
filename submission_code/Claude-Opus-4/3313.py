class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # dp[i][j][0] = max strength using first i elements, j subarrays, not in subarray
        # dp[i][j][1] = max strength using first i elements, j subarrays, currently in subarray
        dp = [[[-float('inf'), -float('inf')] for _ in range(k + 1)] for _ in range(n + 1)]
        
        # Base case: 0 elements, 0 subarrays
        dp[0][0][0] = 0
        
        for i in range(1, n + 1):
            dp[i][0][0] = 0  # Can always choose 0 subarrays
            
            for j in range(1, min(i, k) + 1):
                # Calculate the multiplier for the j-th subarray
                # j=1 -> multiplier = k
                # j=2 -> multiplier = -(k-1)
                # j=3 -> multiplier = k-2
                # Pattern: (-1)^(j+1) * (k - j + 1)
                multiplier = (k - j + 1) * (1 if j % 2 == 1 else -1)
                
                # Case 1: Not including nums[i-1] in any subarray
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1])
                
                # Case 2: Including nums[i-1] in the j-th subarray
                # Option 2a: Start a new subarray at position i
                dp[i][j][1] = dp[i-1][j-1][0] + nums[i-1] * multiplier
                
                # Option 2b: Continue the current subarray
                if dp[i-1][j][1] != -float('inf'):
                    dp[i][j][1] = max(dp[i][j][1], dp[i-1][j][1] + nums[i-1] * multiplier)
        
        return max(dp[n][k][0], dp[n][k][1])