class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix OR array
        prefix_or = [0] * (n + 1)
        for i in range(n):
            prefix_or[i + 1] = prefix_or[i] | nums[i]
        
        # Initialize dp array
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        
        # Fill dp array
        for i in range(1, k + 1):
            for j in range(2 * i, n + 1):
                # Option 1: Don't include the current element
                dp[i][j] = dp[i][j - 1]
                
                # Option 2: Include the current element
                left_or = prefix_or[j - i] ^ prefix_or[j - 2 * i]
                right_or = prefix_or[j] ^ prefix_or[j - i]
                current_value = left_or ^ right_or
                
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 2] + current_value)
        
        return dp[k][n]