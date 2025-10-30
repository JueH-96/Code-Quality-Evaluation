class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        k = target // 2
        m = min(n, k)
        rem = n - m
        
        sum_A = m * (m + 1) // 2
        start_B = target
        sum_B = rem * start_B + (rem * (rem - 1)) // 2
        
        return (sum_A + sum_B) % MOD