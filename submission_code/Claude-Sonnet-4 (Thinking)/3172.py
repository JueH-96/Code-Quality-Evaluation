class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Total sum of all numbers from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Count of numbers divisible by m in range [1, n]
        k = n // m
        
        # Sum of numbers divisible by m
        num2 = m * k * (k + 1) // 2
        
        # num1 - num2 = (total_sum - num2) - num2 = total_sum - 2 * num2
        return total_sum - 2 * num2