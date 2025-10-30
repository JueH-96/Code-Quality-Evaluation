class Solution:
    MOD = 10**9 + 7
    max_fact = 2000
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a = n + k - 1
        b = k
        if a < 0 or b < 0 or a < b:
            return 0
        res = Solution.fact[a] * Solution.inv_fact[b] % Solution.MOD
        res = res * Solution.inv_fact[a - b] % Solution.MOD
        return res