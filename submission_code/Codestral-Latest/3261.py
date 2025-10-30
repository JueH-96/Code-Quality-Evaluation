class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k == 0:
            return functools.reduce(lambda x, y: x | y, nums)

        # Create a 2D DP table where dp[i][j] represents the minimum OR value
        # after applying j operations on the first i elements.
        dp = [[float('inf')] * (k + 1) for _ in range(n)]

        # Initialize the base case
        dp[0][0] = nums[0]

        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] | nums[i]

        for i in range(1, n):
            for j in range(1, k + 1):
                if i - 1 >= 0:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j] | nums[i])
                if i - 1 >= 0 and j - 1 >= 0:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] | (nums[i - 1] & nums[i]))

        return min(dp[n - 1])