class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        total_sum = n * (n + 1) // 2
        k = n // m
        sum_divisible = m * k * (k + 1) // 2
        sum_not_divisible = total_sum - sum_divisible
        return sum_not_divisible - sum_divisible