class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        # dp[i][j] will store the minimum possible sum of values of the subarrays if we consider
        # the first i elements of nums and divide them into j subarrays.
        # Initialize with infinities since we are looking for the minimum.
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 elements, 0 subarrays, sum is 0
        
        # Iterate over each possible number of subarrays from 1 to m
        for j in range(1, m + 1):
            # Iterate over each possible length of nums from 1 to n
            for i in range(j, n + 1):
                # We need to find the optimal previous split point k
                current_and = nums[i - 1]
                # Iterate backwards to find where the current subarray could start
                for k in range(i - 1, j - 2, -1):
                    current_and &= nums[k]
                    # If the AND of this subarray matches the required andValues[j-1]
                    if current_and == andValues[j - 1]:
                        # Update dp[i][j] considering this subarray ending at i-1
                        dp[i][j] = min(dp[i][j], dp[k][j - 1] + nums[i - 1])
        
        # The answer is the minimum sum of values when using all n elements and forming m subarrays
        result = dp[n][m]
        return result if result != float('inf') else -1