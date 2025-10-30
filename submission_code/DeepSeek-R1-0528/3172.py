class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        total_sum = n * (n + 1) // 2
        k = n // m
        sum_divisible = m * (k * (k + 1) // 2)
        return total_sum - 2 * sum_divisible