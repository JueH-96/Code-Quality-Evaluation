class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Sum of all numbers from 1 to n
        total = n * (n + 1) // 2
        
        # Number of multiples of m up to n
        k = n // m
        
        # Sum of all multiples of m: m * (1 + 2 + ... + k) = m * k * (k + 1) // 2
        sum_multiples = m * k * (k + 1) // 2
        
        # num1 = total - sum_multiples, num2 = sum_multiples
        # We want num1 - num2 = (total - sum_multiples) - sum_multiples = total - 2*sum_multiples
        return total - 2 * sum_multiples