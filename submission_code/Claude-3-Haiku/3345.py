class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10 ** 9 + 7
        
        # Create a 2D dynamic programming array to store the number of subsequences with sum equal to k
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Initialize the base case
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(k + 1):
                # Include the current element
                if j >= nums[i - 1]:
                    dp[i][j] = (dp[i - 1][j - nums[i - 1]] + dp[i - 1][j]) % mod
                # Exclude the current element
                else:
                    dp[i][j] = dp[i - 1][j]
        
        # Return the sum of all subsequences with sum equal to k
        return sum(dp[n]) % mod