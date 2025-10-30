class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        sum_total = n * (n + 1) // 2
        k = n // m
        num2 = k * (m + m * k) // 2
        return sum_total - 2 * num2