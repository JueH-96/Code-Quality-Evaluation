class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        mod = 10**9 + 7
        m = min(n, target // 2)
        s1 = m * (m + 1) // 2
        remaining = n - m
        s2 = remaining * (2 * target + remaining - 1) // 2
        return (s1 + s2) % mod