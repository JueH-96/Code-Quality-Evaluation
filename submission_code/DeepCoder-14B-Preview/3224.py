MOD = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: list) -> int:
        # Precompute factorial and inverse factorial up to n
        max_fact = n
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Sort the sick array (though it's already sorted)
        sick_sorted = sorted(sick)
        segments = []
        
        # Process the left segment (before the first sick position)
        if sick_sorted:
            first = sick_sorted[0]
            if first > 0:
                left_start = 0
                left_end = first - 1
                m = left_end - left_start + 1
                segments.append( (1, m) )  # ways=1 for open segment
        
        # Process the middle segments (between consecutive sick positions)
        for i in range(1, len(sick_sorted)):
            prev = sick_sorted[i-1]
            curr = sick_sorted[i]
            seg_start = prev + 1
            seg_end = curr - 1
            if seg_start <= seg_end:
                m = seg_end - seg_start + 1
                ways = pow(2, m-1, MOD)
                segments.append( (ways, m) )
        
        # Process the right segment (after the last sick position)
        if sick_sorted:
            last = sick_sorted[-1]
            if last < n - 1:
                right_start = last + 1
                right_end = n - 1
                m = right_end - right_start + 1
                segments.append( (1, m) )  # ways=1 for open segment
        
        # Calculate product of ways and sum of m_i
        product_ways = 1
        m_list = []
        for way, m in segments:
            product_ways = (product_ways * way) % MOD
            m_list.append(m)
        
        sum_mi = sum(m for _, m in segments)
        if sum_mi == 0:
            return 1 % MOD
        
        # Calculate multinomial coefficient: fact[sum_mi] * product(inv_fact[m_i])
        multinom = fact[sum_mi]
        for m in m_list:
            multinom = multinom * inv_fact[m] % MOD
        
        total = (product_ways * multinom) % MOD
        return total