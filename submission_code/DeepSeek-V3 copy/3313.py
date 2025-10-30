class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Initialize DP table
        # dp[i][j] represents the maximum strength using the first i elements and j subarrays
        # Initialize with -infinity
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(k + 1):
                # If we don't take the current element in any subarray
                if dp[i-1][j] != -float('inf'):
                    dp[i][j] = max(dp[i][j], dp[i-1][j])
                # If we take the current element in a new subarray
                if j > 0:
                    # Determine the multiplier based on the parity of j
                    multiplier = k - j + 1 if j % 2 == 1 else -(k - j + 1)
                    # Try to start a new subarray at the current element
                    # We need to find the best starting point for the new subarray
                    # So we iterate over all possible starting points
                    # But to optimize, we can keep track of the best sum up to the current point
                    # So we need to find the maximum sum of a subarray ending at i-1
                    # We can use a variable to keep track of the maximum sum up to the current point
                    max_sum = -float('inf')
                    current_sum = 0
                    for l in range(i-1, -1, -1):
                        current_sum += nums[l]
                        if dp[l][j-1] != -float('inf'):
                            max_sum = max(max_sum, dp[l][j-1] + current_sum * multiplier)
                    if max_sum != -float('inf'):
                        dp[i][j] = max(dp[i][j], max_sum)
        
        # The answer is the maximum strength using all elements and k subarrays
        return dp[n][k]