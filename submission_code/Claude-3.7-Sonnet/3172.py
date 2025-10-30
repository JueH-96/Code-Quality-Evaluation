class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Sum of all integers from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Calculate how many multiples of m are in the range [1,n]
        k = n // m
        
        # Sum of integers divisible by m (num2)
        sum_divisible = m * k * (k + 1) // 2
        
        # num1 = total_sum - sum_divisible
        # The answer is num1 - num2 which simplifies to:
        return total_sum - 2 * sum_divisible