class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Initialize DP table
        # dp[i][j] represents the maximum strength for the first i elements and j subarrays
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
                    if j % 2 == 1:
                        multiplier = (k - j + 1)
                    else:
                        multiplier = -(k - j + 1)
                    # Iterate over all possible starting points of the current subarray
                    current_sum = 0
                    for l in range(i, 0, -1):
                        current_sum += nums[l-1]
                        if dp[l-1][j-1] != -float('inf'):
                            dp[i][j] = max(dp[i][j], dp[l-1][j-1] + current_sum * multiplier)
        
        # The answer is the maximum value in the last row for j = k
        return dp[n][k]