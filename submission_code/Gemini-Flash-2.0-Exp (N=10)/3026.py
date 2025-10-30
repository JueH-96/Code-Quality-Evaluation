class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        count = min(n, (target + 1) // 2)
        sum1 = (count * (count + 1) // 2) % MOD
        remaining = n - count
        sum2 = ((count + remaining) * (count + remaining + 1) // 2 - count * (count + 1) // 2) % MOD
        return (sum1 + sum2) % MOD