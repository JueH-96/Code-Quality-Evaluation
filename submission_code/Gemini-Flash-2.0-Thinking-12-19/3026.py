class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        inv2 = 500000004
        k = target // 2
        if n <= k - (1 - (target % 2)):
            sum_val = (n * (n + 1)) % MOD
            sum_val = (sum_val * inv2) % MOD
            return sum_val
        else:
            if target % 2 == 0:
                term1 = ((k - 1) * k) % MOD
                term1 = (term1 * inv2) % MOD
                term2 = ((n - k + 1) * target) % MOD
                term3_num = (n - k) * (n - k + 1)
                term3 = (term3_num % MOD) * inv2 % MOD
                sum_val = (term1 + term2 + term3) % MOD
            else:
                term1 = (k * (k + 1)) % MOD
                term1 = (term1 * inv2) % MOD
                term2 = ((n - k) * target) % MOD
                term3_num = (n - k - 1) * (n - k)
                term3 = (term3_num % MOD) * inv2 % MOD
                sum_val = (term1 + term2 + term3) % MOD
            return sum_val