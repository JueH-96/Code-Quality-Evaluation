class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Total sum of integers from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Number of multiples of m in range [1, n]
        k = n // m
        
        # Sum of all multiples of m in range [1, n]
        # m + 2m + 3m + ... + km = m * (1 + 2 + ... + k) = m * k * (k + 1) / 2
        num2 = m * k * (k + 1) // 2
        
        # Sum of integers not divisible by m
        num1 = total_sum - num2
        
        return num1 - num2