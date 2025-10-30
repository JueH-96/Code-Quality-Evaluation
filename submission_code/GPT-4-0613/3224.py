class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        fact = [1]
        for i in range(1, n + 1):
            fact.append(fact[-1] * i % MOD)
        inv = [pow(fact[-1], MOD - 2, MOD)]
        for i in range(n, 0, -1):
            inv.append(inv[-1] * i % MOD)
        inv.reverse()
        def C(n, k):
            if k < 0 or k > n: return 0
            return fact[n] * inv[k] % MOD * inv[n - k] % MOD
        sick = [-1] + sick + [n]
        ans = 1
        for i in range(len(sick) - 1):
            l = sick[i + 1] - sick[i] - 1
            r = sick[i] - sick[i - 1] - 1 if i else l
            ans = ans * C(l + r, l) % MOD * pow(2, max(0, l - 1), MOD) % MOD
        return ans