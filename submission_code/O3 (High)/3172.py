class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        """
        Compute (sum of numbers in [1, n] not divisible by m) - 
               (sum of numbers in [1, n] divisible by m)
        """
        # Total sum of 1..n
        total = n * (n + 1) // 2
        
        # Count of multiples of m in 1..n
        k = n // m
        
        # Sum of multiples of m: m * (1 + 2 + ... + k)
        sum_divisible = m * k * (k + 1) // 2
        
        # Sum of non-divisible numbers
        sum_not_divisible = total - sum_divisible
        
        # Required difference
        return sum_not_divisible - sum_divisible