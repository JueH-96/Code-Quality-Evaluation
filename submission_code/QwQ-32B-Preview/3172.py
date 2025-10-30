class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Calculate the number of multiples of m up to n
        k = n // m
        # Sum of multiples of m
        num2 = m * k * (k + 1) // 2
        # Total sum from 1 to n
        total_sum = n * (n + 1) // 2
        # Difference: total_sum - 2 * num2
        difference = total_sum - 2 * num2
        return difference