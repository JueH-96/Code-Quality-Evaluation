class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        # dp[i][j] = minimum sum when we divide nums[0:i] into j subarrays
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                # Try all possible starting positions for the j-th subarray
                for k in range(j - 1, i):
                    # The j-th subarray is nums[k:i]
                    and_val = nums[k]
                    for idx in range(k + 1, i):
                        and_val &= nums[idx]
                    
                    if and_val == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k][j - 1] + nums[i - 1])
        
        return dp[n][m] if dp[n][m] != float('inf') else -1