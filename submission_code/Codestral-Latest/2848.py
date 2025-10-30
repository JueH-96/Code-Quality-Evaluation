class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Precompute the divisibility relationships
        divisors = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    divisors[i][j] = True

        # dp[mask][last] represents the number of special permutations ending with the number at index 'last'
        dp = [[0] * n for _ in range(1 << n)]

        # Initialize the base case
        for i in range(n):
            dp[1 << i][i] = 1

        # Fill the dp table
        for mask in range(1 << n):
            for last in range(n):
                if mask & (1 << last):
                    for next in range(n):
                        if mask & (1 << next) == 0 and divisors[last][next]:
                            dp[mask | (1 << next)][next] = (dp[mask | (1 << next)][next] + dp[mask][last]) % MOD

        # Sum up all the valid permutations
        result = 0
        for last in range(n):
            result = (result + dp[(1 << n) - 1][last]) % MOD

        return result