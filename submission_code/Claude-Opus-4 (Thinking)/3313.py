class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        INF = float('inf')
        
        # dp[i][j][in_subarray]
        dp = [[[-INF] * 2 for _ in range(k + 1)] for _ in range(n + 1)]
        
        # Base case: processed 0 elements, 0 subarrays, not in subarray
        dp[0][0][0] = 0
        
        # Fill DP table
        for i in range(n):
            for j in range(k + 1):
                # Case 1: Not currently in a subarray (have j complete subarrays)
                if dp[i][j][0] > -INF:
                    # Option 1: Skip current element
                    dp[i + 1][j][0] = max(dp[i + 1][j][0], dp[i][j][0])
                    
                    # Option 2: Start new subarray (the (j+1)-th one)
                    if j < k:
                        subarray_num = j + 1
                        coeff = (-1) ** (subarray_num + 1) * (k - subarray_num + 1)
                        dp[i + 1][j + 1][1] = max(dp[i + 1][j + 1][1], 
                                                   dp[i][j][0] + nums[i] * coeff)
                
                # Case 2: Currently in the j-th subarray
                if j > 0 and dp[i][j][1] > -INF:
                    subarray_num = j
                    coeff = (-1) ** (subarray_num + 1) * (k - subarray_num + 1)
                    
                    # Option 1: Continue current subarray
                    dp[i + 1][j][1] = max(dp[i + 1][j][1], 
                                          dp[i][j][1] + nums[i] * coeff)
                    
                    # Option 2: End current subarray
                    dp[i + 1][j][0] = max(dp[i + 1][j][0], dp[i][j][1])
        
        return dp[n][k][0]