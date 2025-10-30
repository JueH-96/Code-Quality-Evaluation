class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Sum of all numbers from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Count of multiples of m in [1..n]
        k = n // m
        
        # Sum of all multiples of m in [1..n]
        sum_divisible_by_m = m * (k * (k + 1) // 2)
        
        # Difference = (sum of not-divisible) - (sum of divisible)
        # = (total_sum - sum_divisible_by_m) - sum_divisible_by_m
        # = total_sum - 2 * sum_divisible_by_m
        return total_sum - 2 * sum_divisible_by_m