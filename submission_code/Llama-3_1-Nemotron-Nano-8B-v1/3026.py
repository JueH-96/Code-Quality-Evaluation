MOD = 10**9 + 7

class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        if target == 1:
            return (n * (n + 1) // 2) % MOD
        
        k = (target - 1) // 2
        m = k + (1 if target % 2 == 0 else 0)
        
        if n <= m:
            sum_initial = k * (k + 1) // 2
            if target % 2 == 0:
                sum_initial += target // 2
            return sum_initial % MOD
        else:
            t = n - m
            sum_initial = k * (k + 1) // 2
            if target % 2 == 0:
                sum_initial += target // 2
            sum_additional = t * target + t * (t - 1) // 2
            total = (sum_initial + sum_additional) % MOD
            return total