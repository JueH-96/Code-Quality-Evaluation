class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        # Initialize a 3D list to store the number of ways to distribute candies
        dp = [[[0] * (limit + 1) for _ in range(limit + 1)] for _ in range(limit + 1)]

        # Base case: 0 candies can be distributed in exactly 1 way (all get 0 candies)
        dp[0][0][0] = 1

        # Iterate over the number of candies
        for candies in range(1, n + 1):
            # Iterate over the possible number of candies each child can get
            for i in range(limit + 1):
                for j in range(limit + 1):
                    for k in range(limit + 1):
                        # If the sum of candies is equal to the current number of candies
                        if i + j + k == candies:
                            # Increment the number of ways to distribute the candies
                            dp[i][j][k] += (dp[i - 1][j][k] if i > 0 else 0) + \
                                          (dp[i][j - 1][k] if j > 0 else 0) + \
                                          (dp[i][j][k - 1] if k > 0 else 0)

        # Sum up all the ways to distribute n candies
        total_ways = 0
        for i in range(limit + 1):
            for j in range(limit + 1):
                for k in range(limit + 1):
                    if i + j + k == n:
                        total_ways += dp[i][j][k]

        return total_ways