class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Calculate the sum of all numbers from 1 to n
        total_sum = n * (n + 1) // 2
        
        # Calculate the number of multiples of m in the range [1, n]
        k = n // m
        
        # Calculate the sum of all multiples of m
        num2 = m * k * (k + 1) // 2
        
        # Calculate the difference
        difference = total_sum - 2 * num2
        
        return difference