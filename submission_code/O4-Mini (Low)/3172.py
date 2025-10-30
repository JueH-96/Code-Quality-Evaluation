class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Total sum of 1 through n
        total_sum = n * (n + 1) // 2
        # Number of multiples of m up to n
        k = n // m
        # Sum of multiples of m: m * (1 + 2 + ... + k) = m * k * (k + 1) // 2
        sum_multiples = m * k * (k + 1) // 2
        # num1 = total_sum - sum_multiples, num2 = sum_multiples
        # Return num1 - num2 = total_sum - 2 * sum_multiples
        return total_sum - 2 * sum_multiples