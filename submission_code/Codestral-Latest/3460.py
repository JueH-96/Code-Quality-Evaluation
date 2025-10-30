class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7

        # Helper function to calculate the number of ways to arrange `k` inversions in `n` elements
        def count_inversions(n, k):
            if k == 0:
                return 1
            if n == 0:
                return 0
            if k > n * (n - 1) // 2:
                return 0
            dp = [[0] * (k + 1) for _ in range(n + 1)]
            dp[0][0] = 1
            for i in range(1, n + 1):
                dp[i][0] = 1
                for j in range(1, k + 1):
                    dp[i][j] = dp[i - 1][j]
                    if j >= i:
                        dp[i][j] += dp[i - 1][j - i]
                    dp[i][j] %= MOD
            return dp[n][k]

        # Sort requirements by end index
        requirements.sort()

        # Initialize the number of ways to satisfy the requirements
        ways = 1

        # Iterate through each requirement
        for i in range(len(requirements)):
            end, cnt = requirements[i]
            # Calculate the number of ways to satisfy the current requirement
            ways *= count_inversions(end + 1, cnt)
            ways %= MOD

        return ways