class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Sum of all numbers from 1 to n
        total = n * (n + 1) // 2
        # Number of multiples of m in [1, n]
        k = n // m
        # Sum of multiples of m: m + 2m + ... + k*m = m * (1 + 2 + ... + k)
        sum_div = m * k * (k + 1) // 2
        # result = (sum of non-divisible) - (sum of divisible)
        # sum of non-divisible = total - sum_div
        return (total - sum_div) - sum_div