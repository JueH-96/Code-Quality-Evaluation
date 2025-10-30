class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        total = n * (n + 1) // 2
        k = n // m
        sum2 = m * k * (k + 1) // 2
        return total - 2 * sum2