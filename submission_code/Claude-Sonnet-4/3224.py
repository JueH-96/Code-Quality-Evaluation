class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Function to compute modular inverse of factorial
        def mod_inverse(x):
            return pow(x, MOD - 2, MOD)
        
        # Find all gaps (consecutive non-infected segments)
        gaps = []
        
        # Check gap before first sick child
        if sick[0] > 0:
            gaps.append((sick[0], True))  # (size, is_boundary)
        
        # Check gaps between consecutive sick children
        for i in range(1, len(sick)):
            gap_size = sick[i] - sick[i-1] - 1
            if gap_size > 0:
                gaps.append((gap_size, False))  # Internal gap
        
        # Check gap after last sick child
        if sick[-1] < n - 1:
            gaps.append((n - 1 - sick[-1], True))  # (size, is_boundary)
        
        if not gaps:
            return 1
        
        # Total number of children to be infected
        total = sum(size for size, _ in gaps)
        
        # Start with multinomial coefficient
        result = fact[total]
        for size, _ in gaps:
            result = (result * mod_inverse(fact[size])) % MOD
        
        # For each internal gap of size > 1, multiply by 2^(size-1)
        for size, is_boundary in gaps:
            if not is_boundary and size > 1:
                result = (result * pow(2, size - 1, MOD)) % MOD
        
        return result