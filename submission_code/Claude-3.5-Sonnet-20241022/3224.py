class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 1000000007
        
        # Calculate factorials and their modular multiplicative inverses
        fact = [1] * (n + 1)
        inv = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        # Calculate modular multiplicative inverse using Fermat's little theorem
        inv[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n - 1, -1, -1):
            inv[i] = (inv[i + 1] * (i + 1)) % MOD
            
        # Count total non-infected positions
        total = n - len(sick)
        
        # Initialize result with the ways to arrange all non-infected positions
        result = fact[total]
        
        # Process each group of consecutive non-infected positions
        first = sick[0]
        last = n - 1 - sick[-1]
        
        # Handle first group
        if first > 0:
            result = (result * inv[first]) % MOD
            
        # Handle last group
        if last > 0:
            result = (result * inv[last]) % MOD
            
        # Handle middle groups
        for i in range(1, len(sick)):
            gap = sick[i] - sick[i-1] - 1
            if gap > 0:
                # For each middle group, we need to:
                # 1. Divide by factorial of group size (inv[gap])
                # 2. Multiply by 2^(gap-1) for the order of infection within group
                result = (result * inv[gap]) % MOD
                result = (result * pow(2, gap-1, MOD)) % MOD
                
        return result