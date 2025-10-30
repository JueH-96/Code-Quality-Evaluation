class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Identify segments of consecutive non-infected children
        segments = []
        
        # Segment before the first sick child
        if sick[0] > 0:
            segments.append(('boundary', sick[0]))
        
        # Segments between sick children
        for i in range(1, len(sick)):
            if sick[i] - sick[i - 1] > 1:
                segments.append(('middle', sick[i] - sick[i - 1] - 1))
        
        # Segment after the last sick child
        if sick[-1] < n - 1:
            segments.append(('boundary', n - 1 - sick[-1]))
        
        if not segments:
            return 1
        
        total_size = sum(size for _, size in segments)
        
        # Precompute factorials
        fact = [1] * (total_size + 1)
        for i in range(1, total_size + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        def mod_inverse(a):
            return pow(a, MOD - 2, MOD)
        
        # Calculate the number of ways for each segment
        total_ways = 1
        
        for segment_type, size in segments:
            if segment_type == 'boundary':
                ways = 1
            else:  # middle
                ways = pow(2, size - 1, MOD)
            total_ways = (total_ways * ways) % MOD
        
        # Calculate multinomial coefficient
        multinomial = fact[total_size]
        for _, size in segments:
            multinomial = (multinomial * mod_inverse(fact[size])) % MOD
        
        return (total_ways * multinomial) % MOD