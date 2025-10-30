class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Calculate the sum of all integers in the range [1, n]
        total_sum = n * (n + 1) // 2
        
        # Calculate the number of multiples of m in the range [1, n]
        num_multiples = n // m
        
        # Calculate the sum of all multiples of m in the range [1, n]
        sum_multiples = m * num_multiples * (num_multiples + 1) // 2
        
        # Calculate num1 and num2
        num1 = total_sum - sum_multiples
        num2 = sum_multiples
        
        # Return the difference between num1 and num2
        return num1 - num2