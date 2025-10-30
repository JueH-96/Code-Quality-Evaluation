class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        count = min(n, (target + 1) // 2)
        sum1 = (1 + count) * count // 2
        remaining = n - count
        sum2 = (target + (target + remaining - 1)) * remaining // 2
        return (sum1 + sum2) % (10**9 + 7)