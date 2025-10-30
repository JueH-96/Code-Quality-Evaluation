class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        k = (target - 1) // 2
        m = min(n, k)
        sum_A = m * (m + 1) // 2
        remaining = n - m
        
        if remaining == 0:
            return sum_A % MOD
        
        if target % 2 == 0:
            sum_A += target // 2
            remaining -= 1
            if remaining == 0:
                return sum_A % MOD
        
        sum_C = remaining * target + (remaining * (remaining - 1)) // 2
        return (sum_A + sum_C) % MOD