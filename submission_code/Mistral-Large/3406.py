class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7

        # Initialize the dp array
        dp = [[0] * (one + 1) for _ in range(zero + 1)]
        dp[0][0] = 1  # Base case: one way to have 0 zeros and 0 ones

        # Initialize the prefix sum array
        prefix = [[0] * (one + 2) for _ in range(zero + 2)]
        prefix[1][1] = 1  # Base case: one way to have 1 zero and 1 one

        # Fill the dp array
        for i in range(1, zero + 1):
            for j in range(1, one + 1):
                if i > limit and j > limit:
                    dp[i][j] = (prefix[i][j] - prefix[i-limit-1][j] - prefix[i][j-limit-1] + prefix[i-limit-1][j-limit-1]) % MOD
                elif i > limit:
                    dp[i][j] = (prefix[i][j] - prefix[i-limit-1][j]) % MOD
                elif j > limit:
                    dp[i][j] = (prefix[i][j] - prefix[i][j-limit-1]) % MOD
                else:
                    dp[i][j] = prefix[i][j] % MOD

                # Update the prefix sum array
                prefix[i+1][j+1] = (prefix[i+1][j] + prefix[i][j+1] - prefix[i][j] + dp[i][j]) % MOD

        # The result is the sum of all valid dp[i][j]
        result = 0
        for i in range(1, zero + 1):
            for j in range(1, one + 1):
                result = (result + dp[i][j]) % MOD

        return result