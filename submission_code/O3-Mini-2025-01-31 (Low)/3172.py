class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Calculate the total sum from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Count how many multiples of m are there up to n
        k = n // m
        
        # Calculate the sum of multiples of m
        sum_multiples = m * (k * (k + 1) // 2)
        
        # Sum of numbers not divisible by m is total_sum - sum_multiples
        # Difference is (sum of not div by m) - (sum of divisible by m)
        # = total_sum - 2 * sum_multiples
        return total_sum - 2 * sum_multiples