class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7

        # If target is even, we can use all numbers from 1 to n
        if target % 2 == 0:
            return (n * (n + 1) // 2) % MOD

        # If target is odd, we need to avoid pairs that sum to target
        # We can use numbers from 1 to n, but skip the number (target + 1) // 2
        skip = (target + 1) // 2
        return ((n * (n + 1) // 2) - skip) % MOD