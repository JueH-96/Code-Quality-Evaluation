class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Identify groups of non-infected children
        groups = []
        if sick[0] > 0:
            groups.append((0, sick[0] - 1, False))  # Can only be infected from right
        
        for i in range(len(sick) - 1):
            if sick[i + 1] - sick[i] > 1:
                groups.append((sick[i] + 1, sick[i + 1] - 1, True))  # Can be infected from both ends
        
        if sick[-1] < n - 1:
            groups.append((sick[-1] + 1, n - 1, False))  # Can only be infected from left
        
        # Calculate internal ways for each group
        internal_ways = 1
        group_sizes = []
        for start, end, both_ends in groups:
            size = end - start + 1
            group_sizes.append(size)
            if both_ends:
                internal_ways = (internal_ways * pow(2, size - 1, MOD)) % MOD
        
        # Calculate multinomial coefficient
        total_size = sum(group_sizes)
        
        # Precompute factorials
        fact = [1] * (total_size + 1)
        for i in range(1, total_size + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        # Compute inverse factorials
        inv_fact = [1] * (total_size + 1)
        inv_fact[total_size] = pow(fact[total_size], MOD - 2, MOD)
        for i in range(total_size - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
        
        # Compute multinomial coefficient
        multinomial = fact[total_size]
        for size in group_sizes:
            multinomial = (multinomial * inv_fact[size]) % MOD
        
        return (internal_ways * multinomial) % MOD