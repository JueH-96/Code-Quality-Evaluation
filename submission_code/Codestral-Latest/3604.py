class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7

        # Calculate the number of ways to assign performers to stages
        ways_to_assign = pow(x, n, MOD)

        # Calculate the number of ways to assign scores to bands
        ways_to_score = pow(y, x, MOD)

        # The total number of ways is the product of the two
        total_ways = (ways_to_assign * ways_to_score) % MOD

        return total_ways