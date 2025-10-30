class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Sum of all integers from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Number of multiples of m up to n
        k = n // m
        
        # Sum of multiples of m from m to k*m
        sum_multiples = m * (k * (k + 1) // 2)
        
        # num1 = total_sum - sum_multiples, num2 = sum_multiples
        # We want num1 - num2 = (total_sum - sum_multiples) - sum_multiples
        return total_sum - 2 * sum_multiples