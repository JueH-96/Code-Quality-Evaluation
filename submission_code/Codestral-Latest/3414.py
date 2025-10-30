class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7

        # Helper function to perform modular exponentiation
        def mod_exp(base, exp, mod):
            result = 1
            base = base % mod
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % mod
                exp = exp >> 1
                base = (base * base) % mod
            return result

        # Helper function to calculate the number of ways to reach stair k
        def count_ways(k):
            if k == 0:
                return 2
            if k == 1:
                return 4

            # Initialize the number of ways to reach each stair
            dp = [0] * (k + 1)
            dp[0] = 2
            dp[1] = 4

            # Calculate the number of ways to reach each stair
            for i in range(2, k + 1):
                dp[i] = (dp[i - 1] + dp[i - 2]) % MOD
                if i % 2 == 0:
                    dp[i] = (dp[i] + mod_exp(2, i // 2, MOD)) % MOD

            return dp[k]

        return count_ways(k)