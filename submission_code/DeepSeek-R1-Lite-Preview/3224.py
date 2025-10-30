class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        sick = sorted(sick)
        segments = []
        prev = -1
        for s in sick:
            if prev + 1 < s:
                segments.append((s - prev - 1, 1))  # k, m
            prev = s
        if prev + 1 < n:
            segments.append((n - prev - 1, 1))
        
        s = n - len(sick)
        if s == 0:
            return 1
        
        max_fact = s
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i - 1] * i % MOD
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        product_f = 1
        multinom = fact[s]
        for k, m in segments:
            if k == 0:
                continue
            if m == 1:
                f = 1
            else:
                f = pow(2, k - 1, MOD)
            product_f = product_f * f % MOD
            multinom = multinom * inv_fact[k] % MOD
        
        total = product_f * multinom % MOD
        return total