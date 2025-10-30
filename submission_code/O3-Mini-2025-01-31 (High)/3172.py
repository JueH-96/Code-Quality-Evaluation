class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Sum of all numbers from 1 to n
        total = n * (n + 1) // 2
        
        # Count of numbers in [1, n] divisible by m
        k = n // m
        
        # Sum of numbers divisible by m: m * (1 + 2 + ... + k)
        divisible_sum = m * (k * (k + 1) // 2)
        
        # Sum of numbers not divisible by m = total - divisible_sum
        # Difference = (not divisible sum) - (divisible sum)
        return total - 2 * divisible_sum