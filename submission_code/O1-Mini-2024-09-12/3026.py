class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        k = (target - 1) // 2
        m = n - k
        if m < 0:
            # This can happen if n > k + some condition, but in our case, k is always floor((t-1)/2)
            # and n can be up to 10^9, but since target can be up to 10^9, m can be up to 10^9
            # However, m should not be negative because k <= (t-1)/2 <= t
            k = n
            m = 0
        sum_k = (k * (k + 1) // 2) % MOD
        sum_m = (m * target) % MOD
        sum_m += (m * (m - 1) // 2) % MOD
        sum_m %= MOD
        total = (sum_k + sum_m) % MOD
        return total