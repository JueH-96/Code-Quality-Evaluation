class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Total sum of first n natural numbers
        total_sum = n * (n + 1) // 2
        
        # Count of numbers divisible by m in range [1, n]
        k = n // m
        # Sum of numbers divisible by m (they form an arithmetic progression: m, 2m, ... km)
        sum_divisible = m * k * (k + 1) // 2
        
        # Sum of numbers not divisible by m is total_sum - sum_divisible
        # The answer is defined as num1 - num2 = (total_sum - sum_divisible) - sum_divisible
        return total_sum - 2 * sum_divisible