from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        # Initialize a DP table where dp[i][j] represents the minimum sum of subarray values
        # for the first i elements of nums and using the first j elements of andValues.
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: no elements, no subarrays, sum is 0
        
        # Iterate over each position in nums
        for i in range(1, n + 1):
            # Iterate over each subarray count
            for j in range(1, m + 1):
                # Calculate the AND for the subarray ending at i-1
                current_and = nums[i - 1]
                
                # Try to form the j-th subarray ending at position i-1
                for k in range(i, 0, -1):
                    current_and &= nums[k - 1]
                    
                    # If the current AND matches the required andValues[j-1]
                    if current_and == andValues[j - 1]:
                        # Update the DP table
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - 1] + nums[i - 1])
        
        # The answer is the minimum sum for using all m subarrays
        result = dp[n][m]
        
        # If result is still infinity, it means it's not possible to form such subarrays
        return result if result != float('inf') else -1