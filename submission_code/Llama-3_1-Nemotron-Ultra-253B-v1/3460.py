from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        requirements.sort()
        reqs = requirements
        m = len(reqs)
        for i in range(m):
            if reqs[i][0] < i:
                return 0
        
        max_k = max(req[0] for req in reqs) + 1
        max_cnt = max(req[1] for req in reqs)
        
        f = [[0] * (max_cnt + 1) for _ in range(max_k + 1)]
        f[0][0] = 1
        for k in range(1, max_k + 1):
            for c in range(max_cnt + 1):
                for i in range(min(k, c + 1)):
                    f[k][c] = (f[k][c] + f[k-1][c - i]) % MOD
        
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (n + 1)
        inv_fact[n] = pow(fact[n], MOD-2, MOD)
        for i in range(n-1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        def comb(a, b):
            if a < 0 or b < 0 or a < b:
                return 0
            return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD
        
        prev_end = -1
        prev_cnt = 0
        result = 1
        
        for end, cnt in reqs:
            k = end - prev_end
            delta = cnt - prev_cnt
            remaining = n - (prev_end + 1)
            if k > remaining or delta < 0 or delta > (k * (k - 1)) // 2:
                return 0
            ways = comb(remaining, k) * f[k][delta] % MOD
            result = result * ways % MOD
            prev_end = end
            prev_cnt = cnt
        
        return result