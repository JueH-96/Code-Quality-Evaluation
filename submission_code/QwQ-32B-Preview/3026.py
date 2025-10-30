class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        m = target // 2
        
        if n <= m:
            # Sum of first n natural numbers
            total = (n * (n + 1)) // 2
        else:
            # Sum of numbers from 1 to m
            sum1 = (m * (m + 1)) // 2
            # Sum of numbers from target to target + (n - m) - 1
            sum2 = (2 * target + (n - m) - 1) * (n - m) // 2
            total = sum1 + sum2
        
        return total % MOD