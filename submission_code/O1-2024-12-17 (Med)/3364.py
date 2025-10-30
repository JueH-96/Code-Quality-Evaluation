class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        import math
        
        n, m = len(nums), len(andValues)
        # dp[i][j] = minimum sum of the last elements when dividing nums[:i] into j subarrays
        dp = [[math.inf]*(m+1) for _ in range(n+1)]
        dp[0][0] = 0  # Base case: no elements into 0 subarrays => sum = 0
        
        # For each length i of the first segment of nums
        for i in range(1, n+1):
            # We can form at most i subarrays from i elements, but not more than m
            for j in range(1, min(i, m)+1):
                rolling_and = (1 << 17) - 1  # enough bits to cover nums < 10^5
                # We'll try all possible starts t for the j-th subarray (which ends at i-1)
                # subarray = nums[t : i], so t goes from i-1 down to j-1
                for t in range(i-1, j-2, -1):
                    rolling_and &= nums[t]
                    # If at any point the rolling AND falls below andValues[j-1]
                    # it can't match anymore (bits can't come back), so break
                    if rolling_and < andValues[j-1]:
                        break
                    # If we match exactly, update the dp
                    if rolling_and == andValues[j-1]:
                        dp[i][j] = min(dp[i][j], dp[t][j-1] + nums[i-1])
        
        return -1 if dp[n][m] == math.inf else dp[n][m]