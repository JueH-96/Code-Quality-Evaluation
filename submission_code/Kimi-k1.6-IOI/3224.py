class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        if not sick:
            return 0  # As per problem constraints, this case won't occur
        
        # Precompute factorial, inverse factorial, and powers of 2 up to n+2
        max_fact = n
        max_n = max_fact + 2
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        pow2 = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            pow2[i] = pow2[i-1] * 2 % MOD
        
        gaps = []
        # First gap (before first sick)
        first = sick[0]
        if first > 0:
            gaps.append((first, 1))
        
        # Middle gaps (between consecutive sick)
        for i in range(1, len(sick)):
            prev = sick[i-1]
            curr = sick[i]
            m = curr - prev - 1
            if m > 0:
                gaps.append((m, pow2[m-1]))
        
        # Last gap (after last sick)
        last = sick[-1]
        m = (n - 1) - last
        if m > 0:
            gaps.append((m, 1))
        
        if not gaps:
            return 0
        
        sum_m = 0
        product_w = 1
        m_list = []
        for m, w in gaps:
            sum_m += m
            product_w = product_w * w % MOD
            m_list.append(m)
        
        # Compute multinomial coefficient: fact[sum_m] / product(fact[m] for m in m_list)
        numerator = fact[sum_m]
        denominator = 1
        for m in m_list:
            denominator = denominator * fact[m] % MOD
        multinomial = numerator * pow(denominator, MOD-2, MOD) % MOD
        
        return product_w * multinomial % MOD