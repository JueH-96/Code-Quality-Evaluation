class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        fact = [1] * (n + 1)
        inv = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
            inv[i] = pow(fact[i], MOD - 2, MOD)
        ans = 0
        for i in range(n):
            ans = (ans + fact[n - i - 1] * inv[i] * inv[n - i - 1]) % MOD
        return ans