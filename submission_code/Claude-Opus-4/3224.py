class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials
        max_n = n + 1
        fact = [1] * max_n
        for i in range(1, max_n):
            fact[i] = fact[i-1] * i % MOD
        
        # Compute inverse factorials using Fermat's little theorem
        inv_fact = [1] * max_n
        inv_fact[max_n-1] = pow(fact[max_n-1], MOD-2, MOD)
        for i in range(max_n-2, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        # Find segments of non-infected children
        segments = []
        
        # Segment before first sick child
        if sick[0] > 0:
            segments.append((sick[0], False))  # (length, is_middle)
        
        # Segments between sick children
        for i in range(1, len(sick)):
            length = sick[i] - sick[i-1] - 1
            if length > 0:
                segments.append((length, True))  # These are middle segments
        
        # Segment after last sick child
        if sick[-1] < n - 1:
            segments.append((n - 1 - sick[-1], False))
        
        # Calculate total non-infected children
        total_non_infected = sum(seg[0] for seg in segments)
        
        # Calculate result using multinomial coefficient
        result = fact[total_non_infected]
        
        for length, is_middle in segments:
            result = result * inv_fact[length] % MOD
            if is_middle and length > 0:
                # For middle segments, multiply by 2^(length-1)
                result = result * pow(2, length - 1, MOD) % MOD
        
        return result