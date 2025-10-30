class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        mod = 10**9 + 7
        m = n - len(sick)
        if m == 0:
            return 1
        
        max_fact = m
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % mod
        
        seg_lengths = []
        if sick[0] > 0:
            seg_lengths.append(sick[0])
        
        extra = 1
        for i in range(len(sick) - 1):
            gap = sick[i+1] - sick[i] - 1
            if gap > 0:
                seg_lengths.append(gap)
                extra = (extra * pow(2, gap - 1, mod)) % mod
        
        if sick[-1] < n - 1:
            seg_lengths.append(n - 1 - sick[-1])
        
        denom = 1
        for L in seg_lengths:
            denom = (denom * fact[L]) % mod
        
        numerator = fact[m]
        total_ways = numerator * pow(denom, mod - 2, mod) % mod
        total_ways = total_ways * extra % mod
        return total_ways